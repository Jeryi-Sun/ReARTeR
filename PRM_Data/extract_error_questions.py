import argparse
import re
import json
# Sample data to parse
parser = argparse.ArgumentParser(description="Running exp")
parser.add_argument("--dataset_name", type=str)
args = parser.parse_args()
if args.dataset_name == "2wikimultihopqa":
    path = "./2wikimultihopqa/logs/processing_log_11201858.log"
elif args.dataset_name == "hotpotqa":
    path = "./hotpotqa/logs/processing_log_11200945.log"
elif args.dataset_name == "musique":
    path = "./musique/logs/processing_log_11191736.log"
elif args.dataset_name == "strategyqa":
    path = "./strategyqa/logs/processing_log_11211233.log"


with open(path, 'r') as file:
        data = file.read()  # Read file content

data = re.sub(r'^.*HTTP Request: POST https://api2.aigcbest.top/v1/chat/completions "HTTP/1.1 200 OK".*\n?', '', data, flags=re.MULTILINE)

# Regular expression pattern to match required groups
if args.dataset_name == "hotpotqa":
    pattern = re.compile(r"""
        Processed\sProblem\s\d+:                 # Match "Processed Problem" followed by a number and colon
        \s(.+?)\n                                # Capture the question (non-greedy)
        .*?Final\sAnswer:\s(.+?)\n               # Capture final answer within brackets
        #.*?Final\sAnswer:\s\[(.*?)\]             # Capture final answer within brackets
        .*?correctness_flags:\s(\[.*?\])         # Capture correctness_flags including brackets
        .*?mc_score:\s([0-9.]+)                  # Capture mc_score
    """, re.VERBOSE | re.DOTALL)
else:
    pattern = re.compile(r"""
        Processed\sProblem\s\d+:                 # Match "Processed Problem" followed by a number and colon
        \s(.+?)\n                                # Capture the question (non-greedy)
        .*?Final\sAnswer:\s\[(.*?)\]             # Capture final answer within brackets
        .*?correctness_flags:\s(\[.*?\])         # Capture correctness_flags including brackets
        .*?mc_score:\s([0-9.]+)                  # Capture mc_score
    """, re.VERBOSE | re.DOTALL)

# Find all matches in the data
matches = pattern.findall(data)
saved_data = []
for item in matches:
    question = item[0]
    if args.dataset_name == "hotpotqa":
        answer = item[1]
    else:
        answer = eval(item[1])
    mc_score = float(item[3])
    saved_data.append({"problem": question, "final_answer": answer, "mc_score": mc_score})
if args.dataset_name == "2wikimultihopqa":
    saved_data_path = "./2wikimultihopqa/2wikimultihopqa_data_mc.json"
elif args.dataset_name == "hotpotqa":
    saved_data_path = "./hotpotqa/hotpotqa_data_mc.json"
elif args.dataset_name == "musique":
    saved_data_path = "./musique/musique_data_mc.json"
elif args.dataset_name == "strategyqa":
    saved_data_path = "./strategyqa/strategyqa_data_mc.json"

with open(saved_data_path, 'w') as file:
    json.dump(saved_data, file, ensure_ascii=False)

