import os
import re
import torch
import wandb
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

load_dotenv()

# ── config ───────────────────────────────────────────────────────────────────
MODEL_ID   = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
WANDB_PROJ = os.getenv("WANDB_PROJECT", "rlvr-code-correctness")
WANDB_ENT  = os.getenv("WANDB_ENTITY",  "savirpatil-purdue-university")
HF_USER    = os.getenv("HF_USERNAME",   "savirpatil")
OUTPUT_DIR = "./outputs/sft-lora"
SMOKE      = False

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── dataset prep ─────────────────────────────────────────────────────────────
def get_training_data(smoke: bool = False):
    humaneval = load_dataset("openai_humaneval", split="test")
    mbpp      = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")

    def format_humaneval(ex):
        return {
            "text": (
                f"<|im_start|>user\nComplete the following Python function:\n"
                f"{ex['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['canonical_solution']}<|im_end|>"
            )
        }

    def format_mbpp(ex):
        fn_name = ""
        for test in ex["test_list"]:
            match = re.match(r"assert (\w+)\(", test.strip())
            if match:
                fn_name = match.group(1)
                break
        return {
            "text": (
                f"<|im_start|>user\nWrite a Python function named `{fn_name}` that solves the following:\n"
                f"{ex['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['code']}<|im_end|>"
            )
        }

    he_formatted   = humaneval.map(format_humaneval, remove_columns=humaneval.column_names)
    mbpp_formatted = mbpp.map(format_mbpp,           remove_columns=mbpp.column_names)

    combined = concatenate_datasets([he_formatted, mbpp_formatted]).shuffle(seed=42)
    if smoke:
        combined = combined.select(range(8))
    return combined

# ── training ─────────────────────────────────────────────────────────────────
def train(smoke: bool = False):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    if not smoke:
        wandb.init(project=WANDB_PROJ, entity=WANDB_ENT, name="qwen2.5-coder-1.5b-lora-train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    dataset = get_training_data(smoke=smoke)
    print(f"Training samples: {len(dataset)}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1 if smoke else 3,
        per_device_train_batch_size=1 if smoke else 4,
        gradient_accumulation_steps=1 if smoke else 4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        fp16=True,
        logging_steps=1 if smoke else 10,
        save_strategy="epoch",
        report_to="none" if smoke else "wandb",
        run_name="qwen2.5-coder-1.5b-lora-train",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()

    if not smoke:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        model.push_to_hub(f"{HF_USER}/qwen2.5-coder-1.5b-lora-code")
        tokenizer.push_to_hub(f"{HF_USER}/qwen2.5-coder-1.5b-lora-code")
        wandb.finish()
        print(f"Model pushed to HF Hub: {HF_USER}/qwen2.5-coder-1.5b-lora-code")
    else:
        print("Smoke train complete — no save or push in smoke mode")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    train(smoke=args.smoke or SMOKE)