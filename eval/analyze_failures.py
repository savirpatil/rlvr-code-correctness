import os
import sys
import json
import tempfile
import subprocess
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.inference import clean_completion

load_dotenv()

BASE_MODEL_ID = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
LORA_MODEL_ID = os.getenv("LORA_MODEL", "savirpatil/qwen2.5-coder-1.5b-lora-code")
RLVR_MODEL_ID = os.getenv("RLVR_MODEL", "savirpatil/qwen2.5-coder-1.5b-rlvr-code")
N_PROBLEMS    = 30
MAX_NEW       = 512
TEMP          = 0.2

# ── model loading ─────────────────────────────────────────────────────────────
def load_all_models():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024

    print("Loading base...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=torch.float16).to(device).eval()

    print("Loading LoRA...")
    lora_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=torch.float16)
    lora = PeftModel.from_pretrained(lora_base, LORA_MODEL_ID).to(device).eval()

    print("Loading RLVR...")
    rlvr = AutoModelForCausalLM.from_pretrained(RLVR_MODEL_ID, dtype=torch.float16).to(device).eval()

    return {"base": base, "lora": lora, "rlvr": rlvr}, tokenizer, device

# ── generation ────────────────────────────────────────────────────────────────
def generate_one(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return clean_completion(raw)

# ── scoring ───────────────────────────────────────────────────────────────────
def score_completion(prompt, completion, problem):
    """Use the same HumanEval execution format as run_eval.py."""
    code = (
        prompt
        + completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python", fname], timeout=10, capture_output=True, text=True
        )
        return (1.0, "all tests passed") if result.returncode == 0 else (-1.0, "tests failed")
    except subprocess.TimeoutExpired:
        return (-0.5, "timeout")
    finally:
        os.remove(fname)

# ── analysis ──────────────────────────────────────────────────────────────────
def analyze(n_problems=N_PROBLEMS):
    models, tokenizer, device = load_all_models()
    humaneval = list(load_dataset("openai_humaneval", split="test"))[:n_problems]

    results = []
    for i, problem in enumerate(humaneval):
        prompt = problem["prompt"]
        entry  = {"problem_id": i, "prompt": prompt[:120] + "...", "scores": {}, "completions": {}}

        for variant, model in models.items():
            completion       = generate_one(model, tokenizer, prompt, device)
            score, reason    = score_completion(prompt, completion, problem)
            entry["scores"][variant]      = score
            entry["completions"][variant] = completion

        results.append(entry)
        print(f"[{i+1}/{n_problems}] base={entry['scores']['base']:.2f} "
              f"lora={entry['scores']['lora']:.2f} rlvr={entry['scores']['rlvr']:.2f}")

    os.makedirs("results", exist_ok=True)
    with open("results/failure_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    rlvr_wins = [r for r in results if r["scores"]["rlvr"] == 1.0 and r["scores"]["lora"] < 0]
    lora_wins = [r for r in results if r["scores"]["lora"] == 1.0 and r["scores"]["rlvr"] < 0]
    all_fail  = [r for r in results if all(s < 0 for s in r["scores"].values())]
    all_pass  = [r for r in results if all(s == 1.0 for s in r["scores"].values())]

    print(f"\n{'='*60}")
    print(f"RLVR wins (RLVR passes, LoRA fails): {len(rlvr_wins)}")
    print(f"LoRA wins (LoRA passes, RLVR fails): {len(lora_wins)}")
    print(f"All fail:  {len(all_fail)}")
    print(f"All pass:  {len(all_pass)}")

    print(f"\n{'='*60}")
    print("RLVR WINS — problems where RLVR passes but LoRA fails:")
    for r in rlvr_wins[:5]:
        print(f"\n--- Problem {r['problem_id']} ---")
        print(f"Prompt: {r['prompt']}")
        print(f"\n[LoRA]  score={r['scores']['lora']:.2f}")
        print(r["completions"]["lora"][:400])
        print(f"\n[RLVR]  score={r['scores']['rlvr']:.2f}")
        print(r["completions"]["rlvr"][:400])

    print(f"\n{'='*60}")
    print("LORA WINS — problems where LoRA passes but RLVR fails:")
    for r in lora_wins[:5]:
        print(f"\n--- Problem {r['problem_id']} ---")
        print(f"Prompt: {r['prompt']}")
        print(f"\n[LoRA]  score={r['scores']['lora']:.2f}")
        print(r["completions"]["lora"][:400])
        print(f"\n[RLVR]  score={r['scores']['rlvr']:.2f}")
        print(r["completions"]["rlvr"][:400])

    print(f"\nFull results saved to results/failure_analysis.json")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    args = p.parse_args()
    analyze(args.n_problems)