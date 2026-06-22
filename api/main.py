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
    loop = asyncio.get_event_loop()
    
    results = {}
    for variant in ["base", "lora", "rlvr"]:
        try:
            code = await loop.run_in_executor(
                None, lambda v=variant: generate(req.prompt, v, use_chat_template=req.use_chat_template)
            )
            test_lines = [
                line.strip() for line in req.prompt.split("\n")
                if line.strip().startswith("assert")
            ]
            if test_lines:
                # close the docstring if prompt contains an unclosed one, then append completion
                prefix = req.prompt.split("assert")[0].rstrip()
                if prefix.count('"""') % 2 != 0:
                    prefix += '\n    """\n'
                full_code = prefix + code
                print("=== FULL CODE SENT TO REWARD ===")
                print(full_code)
                print("=== TESTS ===")
                print(test_lines)
                score, reason = compute_reward(full_code, test_lines)
            else:
                score, reason = None, "no tests found in prompt"
            results[variant] = {"code": code, "score": score, "reason": reason}
        except Exception as e:
            print(f"ERROR on variant {variant}: {e}")
            results[variant] = {"code": f"Error: {str(e)}", "score": -1.0, "reason": str(e)}

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