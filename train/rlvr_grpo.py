import os
import re
import sys
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

# ── dataset prep ─────────────────────────────────────────────────────────────
def get_training_data(smoke: bool = False):
    """
    GRPO only needs prompts — no labels. We attach test cases as metadata
    so the reward function can execute them at training time to score each
    generated completion.
    """
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
            f"Write a Python function named `{fn_name}` that solves the following:\n"
            f"{ex['prompt']}\n"
            f"Return ONLY the function definition. No explanation, no example usage, no markdown."
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
    """
    GRPOTrainer calls reward_fn(prompts, completions) -> List[float] after
    each generation batch. We close over a prompt->tests lookup so we know
    which test cases to run for each completion without storing them in the
    model input.
    """
    prompt_to_tests = {ex["prompt"]: ex["tests"] for ex in dataset}

    def reward_fn(prompts, completions, **kwargs):
        scores = []
        for prompt, completion in zip(prompts, completions):
            tests = prompt_to_tests.get(prompt, [])
            if not tests:
                scores.append(-1.0)
                continue
            score, _ = compute_reward(completion, tests)
            scores.append(score)
        return scores

    return reward_fn

# ── training ─────────────────────────────────────────────────────────────────
def train(smoke: bool = False):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    if not smoke:
        wandb.init(project=WANDB_PROJ, entity=WANDB_ENT, name="qwen2.5-coder-1.5b-rlvr-train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")

    dataset = get_training_data(smoke=smoke)
    print(f"Training samples: {len(dataset)}")

    reward_fn = make_reward_fn(dataset)

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=4 if smoke else 8,
        max_completion_length=128 if smoke else 512,
        max_steps=8 if smoke else 1000,
        per_device_train_batch_size=4 if smoke else 8,
        gradient_accumulation_steps=1 if smoke else 4,
        learning_rate=1e-5,
        logging_steps=1 if smoke else 10,
        save_steps=500,
        report_to="none" if smoke else "wandb",
        beta=0.1,
        remove_unused_columns=False,
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