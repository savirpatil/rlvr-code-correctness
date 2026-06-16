import os
import re
import tempfile
import subprocess
from typing import List, Tuple

# ── scoring constants ────────────────────────────────────────────────────────
SCORE_ALL_PASS    =  1.0
SCORE_PARTIAL_MAX =  0.5   # partial credit capped here to keep incentive for full pass
SCORE_TIMEOUT     = -0.5
SCORE_ERROR       = -1.0
EXEC_TIMEOUT      =  5     # seconds per subprocess call

# ── reward function ──────────────────────────────────────────────────────────
def compute_reward(code: str, tests: List[str]) -> Tuple[float, str]:
    """
    Execute code against a list of test assert statements and return a
    (score, reason) tuple. This is the core reward signal for GRPO.

    Scoring:
      all tests pass          → +1.0
      n/m tests pass          → (n/m) * 0.5   (partial, capped at 0.5)
      timeout                 → -0.5
      syntax/runtime error    → -1.0
    """
    if not code or not code.strip():
        return SCORE_ERROR, "empty completion"

    test_block = "\n".join(tests)
    full_code  = code + "\n" + test_block

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        fname = f.name

    try:
        result = subprocess.run(
            ["python", fname],
            timeout=EXEC_TIMEOUT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return SCORE_ALL_PASS, "all tests passed"

        # returncode != 0 — run tests individually to get partial credit
        passed = 0
        for test in tests:
            single_code = code + "\n" + test
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
                tf.write(single_code)
                tname = tf.name
            try:
                r = subprocess.run(
                    ["python", tname],
                    timeout=EXEC_TIMEOUT,
                    capture_output=True,
                    text=True,
                )
                if r.returncode == 0:
                    passed += 1
            except subprocess.TimeoutExpired:
                pass
            finally:
                os.remove(tname)

        if passed == 0:
            return SCORE_ERROR, f"0/{len(tests)} tests passed"

        partial = (passed / len(tests)) * SCORE_PARTIAL_MAX
        return partial, f"{passed}/{len(tests)} tests passed"

    except subprocess.TimeoutExpired:
        return SCORE_TIMEOUT, "execution timed out"
    except Exception as e:
        return SCORE_ERROR, f"exception: {str(e)}"
    finally:
        os.remove(fname)


# ── GRPO-compatible wrapper ──────────────────────────────────────────────────
def batch_reward(prompts: List[str], completions: List[str], test_sets: List[List[str]]) -> List[float]:
    """
    Interface expected by GRPOTrainer: takes parallel lists of prompts,
    completions, and test cases and returns a flat list of float scores.
    prompts is passed through but unused — GRPO requires it in the signature.
    """
    return [compute_reward(code, tests)[0] for code, tests in zip(completions, test_sets)]