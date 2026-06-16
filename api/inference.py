import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

BASE_MODEL_ID  = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
LORA_MODEL_ID  = os.getenv("LORA_MODEL",  "savirpatil/qwen2.5-coder-1.5b-lora-code")
RLVR_MODEL_ID  = os.getenv("RLVR_MODEL",  "savirpatil/qwen2.5-coder-1.5b-rlvr-code")
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.2  # lower than training temp for more deterministic inference

# ── model registry ────────────────────────────────────────────────────────────
# We load all three models once at startup and keep them in memory.
# Each variant shares the same tokenizer since they all derive from the same base.
_models    = {}
_tokenizer = None
_device    = None

def get_device():
    if torch.cuda.is_available():    return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def load_models():
    """
    Load all three model variants into memory. Called once at API startup.
    Base and RLVR are full models; LoRA is the base with an adapter applied on top.
    """
    global _models, _tokenizer, _device
    _device = get_device()
    print(f"Loading models on {_device}...")

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.model_max_length = 1024

    # base model
    print("Loading base model...")
    _models["base"] = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=torch.float16, device_map="auto"
    ).eval()

    # LoRA model — load base then apply adapter weights
    print("Loading LoRA model...")
    _lora_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    _models["lora"] = PeftModel.from_pretrained(_lora_base, LORA_MODEL_ID).eval()

    # RLVR model — fully fine-tuned, load directly
    print("Loading RLVR model...")
    _models["rlvr"] = AutoModelForCausalLM.from_pretrained(
        RLVR_MODEL_ID, dtype=torch.float16, device_map="auto"
    ).eval()

    print("All models loaded.")

def clean_completion(completion: str) -> str:
    completion = re.sub(r"```python\s*", "", completion)
    completion = re.sub(r"```\s*", "", completion)
    lines = completion.split("\n")
    cutoff = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("if __name__"):
            cutoff = i
            break
        if i > 0 and (line.startswith("def ") or line.startswith("class ")):
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

def generate(
    prompt: str,
    variant: str,
    use_chat_template: bool = False,
    stream: bool = False,
) -> str:
    """
    Generate a completion from one model variant.
    variant must be one of: 'base', 'lora', 'rlvr'
    use_chat_template=True for MBPP-style natural language prompts.
    """
    if variant not in _models:
        raise ValueError(f"Unknown variant '{variant}'. Must be one of: {list(_models.keys())}")

    model = _models[variant]

    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(formatted, return_tensors="pt").to(_device)
    else:
        inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=_tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    completion = _tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return clean_completion(completion)


def generate_all(prompt: str, use_chat_template: bool = False) -> dict:
    """
    Run all three variants on the same prompt and return results as a dict.
    This is what the API calls for the side-by-side comparison view.
    """
    return {
        variant: generate(prompt, variant, use_chat_template=use_chat_template)
        for variant in ["base", "lora", "rlvr"]
    }