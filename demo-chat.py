import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import MalleableModel, SteeringVector
HF_cache = ""

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Llama-3-8B", device_map='auto', torch_dtype=torch.float16, cache_dir=HF_cache)
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Llama-3-8B", cache_dir=HF_cache)

# 2. Load behavior vector
refusal_behavior_vector = SteeringVector.load('refusal_behavior_vector')

# 3. MalleableModel is a main steering class. Wrap the model with this class first.
malleable_model = MalleableModel(model=model, tokenizer=tokenizer)

# 4. Let's steer the model. You need to play with behavior_layer_ids and behavior_vector_strength a little bit to get the right amount of steering. 
# Once you get the hang of it, it gets really straightforward. 
# behavior_layer_ids is the layers that we steer and behavior_vector_strength is a multiplier to the behavior vector!
malleable_model.steer(
    behavior_vector=refusal_behavior_vector,
    behavior_layer_ids= [15, 16, 17, 18, 19, 20, 21, 22, 23],
    behavior_vector_strength=1.5,
)

# 5. Interactive chat
print("\n=== Chat with Steered Model ===")
print("The model will now refuse all requests.")
print("Type 'quit' to exit.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() == 'quit':
        break
    
    response = malleable_model.respond(prompt)
    print(f"Model: {response}\n")