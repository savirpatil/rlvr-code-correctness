import sys
sys.path.insert(0, ".")
from train.rlvr_grpo import get_training_data, make_reward_fn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

dataset = get_training_data(smoke=True)
prompt = dataset[0]["prompt"]
tests = dataset[0]["tests"]
print("PROMPT:", repr(prompt[:200]))
print("TESTS:", tests)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", dtype=torch.bfloat16)
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=192, do_sample=True, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("RAW COMPLETION:", repr(completion))

reward_fn = make_reward_fn(dataset)
score = reward_fn([prompt], [completion])
print("SCORE:", score)