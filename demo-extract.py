import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import SteeringDataset, SteeringVector
HF_cache = ""

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Llama-3-8B", device_map='auto', torch_dtype=torch.float16, cache_dir=HF_cache)
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Llama-3-8B", cache_dir=HF_cache)

# 2. Load data
with open("docs/demo-data/alpaca.json", 'r') as file:
    alpaca_data = json.load(file)

with open("docs/demo-data/behavior_refusal.json", 'r') as file:
    refusal_data = json.load(file)

questions = alpaca_data['train']
refusal = refusal_data['non_compliant_responses']
compliace = refusal_data['compliant_responses']

# 3. Create our dataset
refusal_behavior_dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=[(item["question"], item["question"]) for item in questions[:100]],
    suffixes=list(zip(refusal[:100], compliace[:100]))
)

# 4. Extract behavior vector for this setup with 8B model, 10000 examples, a100 GPU -> should take around 4 minutes
# To mimic setup from Representation Engineering: A Top-Down Approach to AI Transparency, do method = "pca_diff" amd accumulate_last_x_tokens=1
refusal_behavior_vector = SteeringVector.train(
    model=model,
    tokenizer=tokenizer,
    steering_dataset=refusal_behavior_dataset,
    method="pca_center",
    accumulate_last_x_tokens="suffix-only"
)

# 5. Let's save this behavior vector for later use
refusal_behavior_vector.save('refusal_behavior_vector')