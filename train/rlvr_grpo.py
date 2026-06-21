import os
import re
import sys
import shutil
import torch
import wandb
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reward.reward import compute_reward

load_dotenv()

# ── config ───────────────────────────────────────────────────────────────────
MODEL_ID   = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
WANDB_PROJ = os.getenv("WANDB_PROJECT", "rlvr-code-correctness")
WANDB_ENT  = os.getenv("WANDB_ENTITY",  "savirpatil-purdue-university")
HF_USER    = os.getenv("HF_USERNAME",   "savirpatil")
OUTPUT_DIR = "./outputs/rlvr-grpo"
SMOKE      = False

# ── completion cleaner ───────────────────────────────────────────────────────
def clean_completion(completion: str, fn_name: str = None) -> str:
    completion = re.sub(r"```python", "", completion)
    completion = re.sub(r"```", "", completion)

    if fn_name:
        pattern = rf"def {re.escape(fn_name)}\("
        matches = list(re.finditer(pattern, completion))
        if matches:
            start = matches[-1].start()
            remainder = completion[start:]
            lines = remainder.split("\n")
            cutoff = len(lines)
            for i, line in enumerate(lines[1:], start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                if not line.startswith(" ") and not line.startswith("\t"):
                    cutoff = i
                    break
            return "\n".join(lines[:cutoff]).rstrip()

    lines = completion.split("\n")
    cutoff = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("if __name__"):
            cutoff = i
            break
    lines = lines[:cutoff]
    final_cutoff = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not line.startswith(" ") and not line.startswith("\t"):
            if not any(stripped.startswith(kw) for kw in (
                "def ", "class ", "return", "#", "@", "from ", "import ",
                "if ", "for ", "while ", "try", "with ", "raise", "pass",
                "break", "continue", "else", "elif", "except", "finally", "yield"
            )):
                final_cutoff = i
                break
    return "\n".join(lines[:final_cutoff]).rstrip()

# ── dataset prep ─────────────────────────────────────────────────────────────
def get_training_data(smoke: bool = False):
    humaneval = load_dataset("openai_humaneval", split="test")
    mbpp      = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")

    def format_humaneval(ex):
        test_lines = [
            line.strip() for line in ex["test"].split("\n")
            if line.strip().startswith("assert")
        ]
        return {
            "prompt":  ex["prompt"],
            "tests":   test_lines,
            "dataset": "humaneval",
        }

    def format_mbpp(ex):
        fn_name = ""
        for test in ex["test_list"]:
            match = re.match(r"assert (\w+)\(", test.strip())
            if match:
                fn_name = match.group(1)
                break
        prompt = (
            f"{ex['prompt']}\n"
            f"def {fn_name}(...):\n"
            f"Complete this function. Output ONLY the code, starting directly with 'def {fn_name}'. "
            f"Do not include explanations, markdown fences, or multiple versions."
        )
        return {
            "prompt":  prompt,
            "tests":   ex["test_list"],
            "dataset": "mbpp",
        }

    he   = humaneval.map(format_humaneval, remove_columns=humaneval.column_names)
    mbpp = mbpp.map(format_mbpp,           remove_columns=mbpp.column_names)

    combined = concatenate_datasets([he, mbpp]).shuffle(seed=42)
    if smoke:
        combined = combined.select(range(8))
    return combined

# ── reward wrapper ────────────────────────────────────────────────────────────
def make_reward_fn(dataset):
    prompt_to_tests = {ex["prompt"]: ex["tests"] for ex in dataset}
    prompt_to_fn = {}
    for ex in dataset:
        match = re.search(r"def (\w+)\(", ex["prompt"])
        prompt_to_fn[ex["prompt"]] = match.group(1) if match else None

    def reward_fn(prompts, completions, **kwargs):
        scores = []
        for prompt, completion in zip(prompts, completions):
            tests = prompt_to_tests.get(prompt, [])
            fn_name = prompt_to_fn.get(prompt)
            if not tests:
                scores.append(-1.0)
                continue
            cleaned = clean_completion(completion, fn_name=fn_name)
            score, _ = compute_reward(cleaned, tests)
            scores.append(score)
        return scores

    return reward_fn

# ── training ─────────────────────────────────────────────────────────────────
def train(smoke: bool = False):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    disk = shutil.disk_usage("/kaggle/working") if os.path.exists("/kaggle/working") else shutil.disk_usage(".")
    print(f"Disk free: {disk.free / 1e9:.1f} GB")

    if not smoke:
        wandb.init(project=WANDB_PROJ, entity=WANDB_ENT, name="qwen2.5-coder-1.5b-rlvr-train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

    dataset = get_training_data(smoke=smoke)
    print(f"Training samples: {len(dataset)}")

    reward_fn = make_reward_fn(dataset)

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=4 if smoke else 4,
        max_completion_length=128 if smoke else 192,
        temperature=0.8,
        max_steps=8 if smoke else 80,
        per_device_train_batch_size=4 if smoke else 4,
        gradient_accumulation_steps=1 if smoke else 2,
        learning_rate=1e-5,
        logging_steps=1 if smoke else 5,
        save_strategy="no",
        report_to="none" if smoke else "wandb",
        beta=0.1,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
    )

    if smoke:
        print("Smoke init complete — skipping train() on MPS (OOM by design, run on Kaggle)")
    else:
        trainer.train()

        if os.path.exists(OUTPUT_DIR):
            for ckpt in os.listdir(OUTPUT_DIR):
                if ckpt.startswith("checkpoint-"):
                    shutil.rmtree(os.path.join(OUTPUT_DIR, ckpt), ignore_errors=True)

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        model.push_to_hub(f"{HF_USER}/qwen2.5-coder-1.5b-rlvr-code")
        tokenizer.push_to_hub(f"{HF_USER}/qwen2.5-coder-1.5b-rlvr-code")
        wandb.finish()
        print(f"Model pushed to HF Hub: {HF_USER}/qwen2.5-coder-1.5b-rlvr-code")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    train(smoke=args.smoke or SMOKE)