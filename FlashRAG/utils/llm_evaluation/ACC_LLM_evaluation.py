import json
import argparse
import os
import requests
import time
from tqdm import tqdm
# os.environ['http_proxy'] = "http://10.74.176.8:11080"
# os.environ['https_proxy'] = "http://10.74.176.8:11080"

url = "https://api2.aigcbest.top/v1/chat/completions"

parser = argparse.ArgumentParser(description="Running exp")
parser.add_argument("--path_dir", type=str)
args = parser.parse_args()



file_path = os.path.join(args.path_dir, 'intermediate_data.json')
with open(file_path, 'r') as f:
    data = json.load(f)

prompt = """Given a question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the golden_answer. Respond with True if the prediction is correct and False otherwise.
Question: {}
Golden Answer: {}
Predicted Answer: {}"""


def judge(question, golden_answer, pred, retries=3):
    gpt_prompt = prompt.format(question, golden_answer, pred)
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": gpt_prompt
            }
        ]
    })
    
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer iDa32s4C2N4gkvThsEezIjhicE1bm4fDeXl0h0ZH3fsPArTi',#MKaSQPD7q85FT2UuswoSE7dnNyOrtR7CMwkxCCp6e8uir3OZ',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    for attempt in range(retries):
        try:
            time.sleep(1)
            response = requests.request("POST", url, headers=headers, data=payload)
            result = response.json()
            print(result)
            result_text = result['choices'][0]['message']['content'].strip()
            return result_text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed. Retrying...")
            time.sleep(1)
            print(e)


    raise ValueError("Failed to get a valid response after multiple retries")

pred_result_list = []
for item in tqdm(data):
    print(file_path)
    question = item["question"]
    golden_answers_list = item["golden_answers"] if isinstance(item["golden_answers"], list) else [item["golden_answers"]]
    pred = item["output"]["pred"]
    pred_item_True = False
    for golden_answer in golden_answers_list:
        if "true" in judge(question, golden_answer, pred).lower():
            pred_item_True = True
            break
    if pred_item_True:
        pred_result_list.append(1)
    else:
        pred_result_list.append(0)


acc = sum(pred_result_list)/len(pred_result_list)


save_file_path = os.path.join(args.path_dir, "metric_score.txt")

# Read the content of the file
with open(save_file_path, "r") as file:
    lines = file.readlines()

# Append the new metric
lines.append(f"acc_llm: {acc}\n")

# Write back to the file
with open(save_file_path, "w") as file:
    file.writelines(lines)




