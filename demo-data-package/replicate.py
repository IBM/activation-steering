import json
import os
import random
from genai import Client, Credentials
from genai.schema import DecodingMethod, HumanMessage, SystemMessage, TextGenerationParameters
from tqdm import tqdm

# Set up environment variables
PATH = '/dccstor/bruce-research/cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

# Set up IBM BAM API client
credentials = Credentials(api_key="pak-4HzllqFzW5SllaYDlFE-NEYkzqWe9QPI9-aQqVXIqf4", api_endpoint="https://bam-api.res.ibm.com")
client = Client(credentials=credentials)

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def generate_questions(prompt, num_questions=10):
    parameters = TextGenerationParameters(
        decoding_method=DecodingMethod.GREEDY,
        max_new_tokens=3000
    )
    
    response = client.text.chat.create(
        model_id="mistralai/mixtral-8x7b-instruct-v01",
        messages=[
            HumanMessage(content=prompt),
        ],
        parameters=parameters,
    )
    
    generated_text = response.results[0].generated_text

    question = ""
    start_tag = f"<start>"
    end_tag = f"</end>"
    start_index = generated_text.find(start_tag)
    end_index = generated_text.find(end_tag)
    #print(start_index)
    if start_index != -1 and end_index != -1:
        question = generated_text[start_index + len(start_tag):end_index].strip()
    
    return question

def create_prompt(target, paraphrased):
    prompt = f"""You are a paraphraser that only changes one ~ three words. I will present two sentences Base and Paraphrased. Your job is to write Paraphrased again and change a one or two words.

BASE\n{target}\n\n

PARAPHRASED \n{paraphrased}\n\n

enclose you paraphrased version with <start> and </end>"""

    return prompt


def process_json(input_file, output_file):
    # Read the input JSON file
    data = read_json_file(input_file)
    
    # Process each item in the 'train' list
    for item in tqdm(data['train'], desc="Processing items"):
        base = item['base']
        for key, paraphrased in item.items():
            if key != 'base':
                prompt = create_prompt(base, paraphrased)
                new_paraphrased = generate_questions(prompt)
                item[key] = new_paraphrased
        break

    for item in tqdm(data['test'], desc="Processing items"):
        base = item['base']
        for key, paraphrased in item.items():
            if key != 'base':
                prompt = create_prompt(base, paraphrased)
                new_paraphrased = generate_questions(prompt)
                item[key] = new_paraphrased
        break
    
    # Write the updated data to the output JSON file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=2)

# Usage
input_file = 'condition_multiple.json'  # Replace with your input file path
output_file = 'condition_multiple_o.json'  # Replace with your desired output file path
process_json(input_file, output_file)