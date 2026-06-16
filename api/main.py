import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.inference import load_models, generate_all, generate
from reward.reward import compute_reward

load_dotenv()

app = FastAPI(title="RLVR Code Correctness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    # run model loading in a thread so it doesn't block the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)

# ── schemas ───────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    use_chat_template: bool = False  # True for natural language prompts (MBPP-style)

class ScoreRequest(BaseModel):
    code: str
    tests: list[str]

class GenerateOneRequest(BaseModel):
    prompt: str
    variant: str          # 'base' | 'lora' | 'rlvr'
    use_chat_template: bool = False

# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    """
    Run all three model variants on the prompt and return completions + reward scores.
    This is the main endpoint the frontend calls for side-by-side comparison.
    """
    loop = asyncio.get_event_loop()

    # generate from all three variants (blocking, run in thread)
    completions = await loop.run_in_executor(
        None, lambda: generate_all(req.prompt, use_chat_template=req.use_chat_template)
    )

    # score each completion against any assert statements found in the prompt
    # (for HumanEval-style prompts that include doctest examples)
    test_lines = [
        line.strip() for line in req.prompt.split("\n")
        if line.strip().startswith("assert")
    ]

    results = {}
    for variant, code in completions.items():
        if test_lines:
            score, reason = compute_reward(code, test_lines)
        else:
            score, reason = None, "no tests found in prompt"
        results[variant] = {
            "code":   code,
            "score":  score,
            "reason": reason,
        }

    return {"results": results}

@app.post("/generate/one")
async def generate_one_endpoint(req: GenerateOneRequest):
    """Generate from a single variant — used for streaming one model at a time."""
    loop = asyncio.get_event_loop()
    code = await loop.run_in_executor(
        None, lambda: generate(req.prompt, req.variant, req.use_chat_template)
    )
    return {"variant": req.variant, "code": code}

@app.post("/score")
async def score_endpoint(req: ScoreRequest):
    """
    Score arbitrary code against provided test cases.
    Used by the frontend when the user supplies their own tests.
    """
    loop = asyncio.get_event_loop()
    score, reason = await loop.run_in_executor(
        None, lambda: compute_reward(req.code, req.tests)
    )
    return {"score": score, "reason": reason}