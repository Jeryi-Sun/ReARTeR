import os
import json
from collections import defaultdict
import argparse
import json
from flashrag.prompt.reasoning_examplars import REASON_PROMPT

# Sample data to parse
parser = argparse.ArgumentParser(description="Running exp")
parser.add_argument("--dataset_name", type=str)
args = parser.parse_args()

# Directory path containing JSON files
directory_path = f'./{args.dataset_name}/results/'

# Initialize a dictionary to store merged data
merged_data = []

# Define priority for types
type_priority = {'leaf': 3, 'best': 2, 'add': 1}
all_data = 0
# Load and process each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            for entry in data:
                question = entry['question']
                partial_answer = tuple(entry['partial_answer'])
                current_type = entry['type']
                current_mc_score = entry['mc_score']
                merged_data.append([question, partial_answer, {'mc_score': current_mc_score, 'type': current_type}])
                

# Post-process to calculate the average mc_score and format data
final_data = []
train_reward_data = []
for question, partial_answer, info in merged_data.items():
    avg_mc_score = info['mc_score']  # Average mc_score
    final_entry = {
        'question': question,
        'partial_answer': list(partial_answer),
        'mc_score': avg_mc_score,
        'type': info['type']
    }
    final_data.append(final_entry)
    user_prompt = (
    REASON_PROMPT+
    f"#\nQuestion: {question}\n"
    )
    messages = [
        {
        "role": "user",
        "content":  user_prompt
        },
        {
        "role": "assistant",
        "content":  "".join(list(partial_answer))
        }]
    kto_entry = {"messages": messages, "label": True if avg_mc_score>0.5 else False}
    train_reward_data.append(kto_entry)



# Save the merged data to a new JSON file
output_file = os.path.join(directory_path, 'reward_data/merged_results.json')
with open(output_file, 'w') as outfile:
    json.dump(final_data, outfile, indent=4)

output_file = os.path.join(directory_path, 'reward_data/rag_reward_data.json')
with open(output_file, 'w') as outfile:
    json.dump(train_reward_data, outfile, indent=4)