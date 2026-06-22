# RLVR Code Correctness 🧠

A live coding assistant that compares three variants of Qwen2.5-Coder-1.5B — base, LoRA SFT, and GRPO-trained RLVR — side by side with real-time unit test scoring. Built to answer a concrete research question about whether RL training produces qualitatively different solution-exploration behavior from supervised fine-tuning.

## How it works

```
Coding problem prompt (HumanEval-style or natural language)
       ↓
FastAPI backend — runs prompt through all three model variants in sequence
    ├── Base Model ──────────────────┐
    │   Qwen2.5-Coder-1.5B-Instruct  ├── same prompt, independent generations
    ├── LoRA SFT ────────────────────┤
    │   fine-tuned on HumanEval +    │
    │   MBPP canonical solutions     │
    └── RLVR (GRPO) ────────────────┘
        trained with binary unit test rewards
        4 completions per prompt, relative reward scoring
        reward signal: +1.0 all pass / partial / -0.5 timeout / -1.0 error
       ↓
Reward scorer — executes completions against extracted assert statements
    subprocess isolation · 5s timeout · partial credit scoring
       ↓
Side-by-side UI — three panels with live scores and pass/fail badges
```

## Quick start

Requirements: Python 3.11+

```bash
git clone https://github.com/savirpatil/rlvr-code-correctness
cd rlvr-code-correctness
conda create -n rlvr python=3.11 -y && conda activate rlvr
pip install -r requirements.txt
```

Create `.env`:

```
HF_TOKEN=
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_PROJECT=rlvr-code-correctness
BASE_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
LORA_MODEL=savirpatil/qwen2.5-coder-1.5b-lora-code
RLVR_MODEL=savirpatil/qwen2.5-coder-1.5b-rlvr-code
HF_USERNAME=savirpatil
```

Launch:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `frontend/index.html` in your browser, enter a coding problem, and watch all three models generate and score in real time.

## Models

| Variant | HuggingFace | Training |
|---------|-------------|----------|
| Base | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | — |
| LoRA SFT | `savirpatil/qwen2.5-coder-1.5b-lora-code` | 3 epochs, HumanEval + MBPP canonical solutions |
| RLVR (GRPO) | `savirpatil/qwen2.5-coder-1.5b-rlvr-code` | 80 steps, binary unit test rewards |

## Results

**HumanEval**

| Model | pass@1 | pass@2 | pass@5 | pass@10 |
|-------|--------|--------|--------|---------|
| Base | 0.508 | 0.627 | 0.758 | 0.829 |
| LoRA SFT | 0.486 | 0.610 | 0.740 | 0.799 |
| RLVR (GRPO) | 0.491 | 0.611 | 0.744 | 0.817 |

**MBPP**

| Model | pass@1 | pass@2 | pass@5 | pass@10 |
|-------|--------|--------|--------|---------|
| Base | 0.429 | 0.511 | 0.597 | 0.646 |
| LoRA SFT | 0.312 | 0.446 | 0.577 | 0.642 |
| RLVR (GRPO) | 0.447 | 0.519 | 0.601 | 0.642 |

W&B report: [rlvr-code-correctness](https://wandb.ai/savirpatil-purdue-university/rlvr-code-correctness)

## Key findings

**LoRA SFT hurts MBPP pass@1 significantly** (0.429 → 0.312) while RLVR exceeds base (0.447). Both models trained on identical data — the difference is the training signal. LoRA learned to imitate the surface pattern of canonical solutions; RLVR optimized for correctness and generalized better across prompt styles.

**RLVR shows higher pass@10 than LoRA on HumanEval** (0.817 vs 0.799) despite similar pass@1, consistent with the hypothesis that reward-driven training generates more diverse correct solutions rather than just shifting the modal output.

**Qualitative behavioral difference**: In live inference, LoRA occasionally produces logically flawed algorithms that pattern-match the structure of a correct solution without implementing the right logic — e.g. checking for duplicate values instead of complement pairs on two-sum. RLVR and base do not exhibit this failure mode on the same prompts.

## Engineering decisions

**Execution-based reward function** — completions are written to a tempfile and executed in a restricted subprocess with a 5-second timeout. Partial credit scoring (n/m × 0.5) prevents the reward from collapsing to binary signal on multi-assert problems, keeping the GRPO gradient informative throughout training.

**bfloat16 for GRPO training** — float16 causes NaN in the probability tensor during generation on GRPO's multi-completion sampling step due to limited dynamic range. bfloat16 has the same memory footprint with a larger exponent range, resolving the instability without increasing GPU memory usage.

**Completion cleaning before reward scoring** — instruct models respond conversationally to code prompts, emitting markdown fences, prose explanations, and stub-then-solution patterns. The cleaner finds the last occurrence of `def {fn_name}(` in the output and extracts that block, which is the model's final and most complete attempt. This was the critical fix that unblocked the reward signal from being permanently stuck at -1.

**save_strategy="no" in GRPOConfig** — disabling mid-training checkpoints on Kaggle's free tier avoids filling the 20GB working directory with 3GB model shards during training, leaving space for the final push to HF Hub.

## Reward scoring

```
all tests pass     → +1.0
n/m tests pass     → (n/m) × 0.5   (partial, capped to incentivize full pass)
timeout (5s)       → -0.5
syntax/runtime err → -1.0
```

Common imports (`math`, `re`, `typing`) are auto-injected before execution since models frequently use standard library functions without importing them.

## Training

**LoRA SFT** — TRL SFTTrainer, r=16, lora_alpha=32, target_modules=[q_proj, v_proj], 3 epochs, lr=2e-4 cosine, effective batch 16. Run on Kaggle T4 x2.

**RLVR (GRPO)** — TRL GRPOTrainer, 4 generations per prompt, max_completion_length=192, beta=0.1 KL penalty, 80 steps, lr=1e-5, bfloat16, gradient checkpointing. Run on Kaggle T4 x2.

## Project structure

```
rlvr-code-correctness/
├── reward/
│   ├── reward.py             # execution-based reward function, partial credit scoring
│   └── sandbox_test.py       # 20-case unit test suite for the reward function itself
├── eval/
│   ├── run_eval.py           # HumanEval + MBPP eval, pass@k, W&B logging
│   └── analyze_failures.py   # qualitative failure auditing across model variants
├── train/
│   ├── sft_lora.py           # LoRA SFT training script
│   └── rlvr_grpo.py          # GRPO training script with reward wrapper
├── api/
│   ├── main.py               # FastAPI, per-variant generation, live reward scoring
│   └── inference.py          # model loading, clean_completion, generate_all
├── frontend/
│   └── index.html            # side-by-side UI, live scores, pass/fail badges
└── notebooks/
    ├── kaggle_sft.ipynb      # Kaggle training notebook for LoRA SFT
    └── kaggle_rlvr.ipynb     # Kaggle training notebook for GRPO
```

## Tech stack

Qwen2.5-Coder · TRL (SFTTrainer + GRPOTrainer) · PEFT · HuggingFace · FastAPI · W&B · Kaggle T4