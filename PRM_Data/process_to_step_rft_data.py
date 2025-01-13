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
directory_path = f'./{args.dataset_name}/rft_results/'

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
                all_data+=len(data)
                question = entry['question']
                partial_answer = tuple(entry['partial_answer'])
                current_type = entry['type']
                current_mc_score = entry['mc_score']
                
  
                merged_data.append([question, partial_answer, {'mc_score': current_mc_score, 'type': current_type}])
                


# Post-process to calculate the average mc_score and format data
final_data = []
train_reward_data = []
rft_data = []
for (question, partial_answer), info in merged_data.items():
    avg_mc_score = sum(info['mc_score']) / len(info['mc_score'])  # Average mc_score
    user_prompt = REASON_PROMPT+f"#\nQuestion: {question}"
    reason_data_list = []
    reason_data_dict_cache = None
    for idx in [len(partial_answer)-1]:
        if "Search: No" in partial_answer[idx] or "So the final answer is:" in partial_answer[idx]:
            messages = [
                {
                "role": "user",
                "content":  user_prompt+"\n".join(partial_answer[:idx]),
                },
                {
                "role": "assistant",
                "content":  partial_answer[idx]
                }]
            kto_entry = {"messages": messages, "label": True if avg_mc_score>0.4 else False}
            reason_data_list.append(kto_entry)
            
        else:

            part_1 = partial_answer[idx].split('Retrieved documents:', maxsplit=1)[0].strip()
            messages = [
                {
                "role": "user",
                "content":  user_prompt+"\n".join(partial_answer[:idx]),
                },
                {
                "role": "assistant",
                "content":  part_1
                }]
            kto_entry = {"messages": messages, "label": True if avg_mc_score>0.4 else False}

            reason_data_list.append(kto_entry)

            part_1 = partial_answer[idx].split('Intermediate answer:', maxsplit=1)[0].strip()
            part_2 = partial_answer[idx].split('Intermediate answer:', maxsplit=1)[1].strip()
            messages = [
                {
                "role": "user",
                "content":  user_prompt+"\n".join(partial_answer[:idx])+'\n'+part_1+'\n'+"Intermediate answer:",
                },
                {
                "role": "assistant",
                "content":  part_2
                }]
            kto_entry = {"messages": messages, "label": True if avg_mc_score>0.4 else False}
            reason_data_list.append(kto_entry)
            

    rft_data += reason_data_list


import pdb
pdb.set_trace()
output_file = os.path.join(directory_path, 'reward_data/rft_data.json')
with open(output_file, 'w') as outfile:
    json.dump(rft_data, outfile, indent=4)