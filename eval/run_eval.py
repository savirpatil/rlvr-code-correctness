import os
import re
import wandb
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from typing import List
import tempfile
import subprocess

load_dotenv()

# ── config ──────────────────────────────────────────────────────────────────
MODEL_ID   = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
WANDB_PROJ = os.getenv("WANDB_PROJECT", "rlvr-code-correctness")
WANDB_ENT  = os.getenv("WANDB_ENTITY",  "savirpatil-purdue-university")
N_SAMPLES  = 10
MAX_NEW    = 512
TEMP       = 0.8
SMOKE      = False

# smoke overrides
SMOKE_N_SAMPLES = 2
SMOKE_MAX_NEW   = 256
SMOKE_TEMP      = 0.2
SMOKE_PROBLEMS  = 2
SMOKE_TIMEOUT   = 15

# ── pass@k estimator ─────────────────────────────────────────────────────────
def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))

# ── completion cleaner ───────────────────────────────────────────────────────
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
    # strip trailing prose: unindented lines that are natural language, not code keywords
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

# ── correctness checkers ─────────────────────────────────────────────────────
def check_correctness(problem: dict, completion: str, timeout: int = 5) -> bool:
    """HumanEval: wraps completion with prompt + test harness using check()."""
    code = (
        problem["prompt"]
        + completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})"
    )
    if os.getenv("DEBUG_EXEC"):
        print("=== EXECUTING ===")
        print(code)
        print("=== END ===")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python", fname], timeout=timeout, capture_output=True, text=True
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.remove(fname)

def check_correctness_mbpp(problem: dict, completion: str, timeout: int = 5) -> bool:
    """MBPP: tests are raw assert statements, no check() wrapper needed."""
    code = completion + "\n" + problem["test"]
    if os.getenv("DEBUG_EXEC"):
        print("=== EXECUTING (MBPP) ===")
        print(code)
        print("=== END ===")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python", fname], timeout=timeout, capture_output=True, text=True
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        if os.getenv("DEBUG_TIMEOUT"):
            print(f"  TIMEOUT after {timeout}s")
        return False
    finally:
        os.remove(fname)

# ── generation ───────────────────────────────────────────────────────────────
def generate_completions(model, tokenizer, prompt: str, n: int, device: str, use_chat_template: bool = False) -> List[str]:
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=True,
            temperature=TEMP,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return [
        clean_completion(tokenizer.decode(o[prompt_len:], skip_special_tokens=True))
        for o in outputs
    ]

# ── eval loop ────────────────────────────────────────────────────────────────
def evaluate(model, tokenizer, dataset, name: str, device: str, run_name: str, checker=None, timeout: int = 5, use_chat_template: bool = False):
    if checker is None:
        checker = check_correctness
    results = []
    problems = list(dataset)
    if SMOKE:
        problems = problems[:SMOKE_PROBLEMS]

    for i, problem in enumerate(problems):
        completions = generate_completions(model, tokenizer, problem["prompt"], N_SAMPLES, device, use_chat_template=use_chat_template)
        correct = sum(checker(problem, c, timeout=timeout) for c in completions)
        results.append((N_SAMPLES, correct))
        if SMOKE or (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{len(problems)} done")

    metrics = {}
    for k in [1, 2, 5, 10]:
        if k <= N_SAMPLES:
            scores = [pass_at_k(n, c, k) for n, c in results]
            metrics[f"pass@{k}"] = float(np.mean(scores))

    print(f"\n{name} results:", metrics)
    return metrics

# ── main ─────────────────────────────────────────────────────────────────────
def main(run_name: str = "qwen2.5-coder-1.5b-base-eval", model_path: str = None):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    path = model_path or MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="auto")
    model.eval()

    humaneval = load_dataset("openai_humaneval", split="test")
    mbpp      = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    def normalize_mbpp(ex):
        # extract expected function name from first assert so prompt is explicit
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
            "prompt":      prompt,
            "test":        "\n".join(ex["test_list"]),
            "entry_point": fn_name,
            "code":        ex["code"],
        }
    mbpp = mbpp.map(normalize_mbpp)

    if not SMOKE:
        run = wandb.init(project=WANDB_PROJ, entity=WANDB_ENT, name=run_name)

    if SMOKE:
        global N_SAMPLES, MAX_NEW, TEMP
        N_SAMPLES = SMOKE_N_SAMPLES
        MAX_NEW   = SMOKE_MAX_NEW
        TEMP      = SMOKE_TEMP
        eval_timeout = SMOKE_TIMEOUT
    else:
        eval_timeout = 5

    he_metrics   = evaluate(model, tokenizer, humaneval, "HumanEval", device, run_name, timeout=eval_timeout)
    mbpp_metrics = evaluate(model, tokenizer, mbpp,      "MBPP",      device, run_name, checker=check_correctness_mbpp, timeout=eval_timeout, use_chat_template=True)

    all_metrics = {f"humaneval/{k}": v for k, v in he_metrics.items()}
    all_metrics.update({f"mbpp/{k}": v for k, v in mbpp_metrics.items()})

    if not SMOKE:
        wandb.log(all_metrics)
        run.finish()

    return all_metrics

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-name",   default="qwen2.5-coder-1.5b-base-eval")
    p.add_argument("--model-path", default=None)
    p.add_argument("--smoke",      action="store_true")
    args = p.parse_args()
    if args.smoke:
        SMOKE = True
    main(args.run_name, args.model_path)