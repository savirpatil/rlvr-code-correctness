import sys
sys.path.insert(0, ".")
from train.rlvr_grpo import clean_completion
from reward.reward import compute_reward

raw = ' \nTo ensure the code is self-contained and clear, you may want to add some comments within the function body.\n\n```python\n# Add your code here\n```\n```python\ndef sector_area(radius, angle):\n    if angle > 360:\n        return None\n    # Calculate the area of the sector\n    area = (angle / 360) * math.pi * (radius ** 2)\n    return area\n```\nIn the solution provided...'

cleaned = clean_completion(raw, fn_name="sector_area")
print("CLEANED:", repr(cleaned))

tests = ['assert sector_area(4,45)==6.283185307179586']
score, reason = compute_reward(cleaned, tests)
print("SCORE:", score, reason)
