# score_list 中计算 original score

import json
from flashrag.prompt.reasoning_examplars import REASON_PROMPT, EXP_PROMPT, CORRECT_PROMPT, EXP_SYS_PROMPT, SUMMATY_PROMPT
from tqdm import tqdm
import numpy as np
data = []
with open("./musique/intermediate_data.json", 'r') as f:
    data += json.load(f)
with open("./hotpotqa/intermediate_data.json", 'r') as f:
    data += json.load(f)

with open("./2wikimultihopqa/item_immediate.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))
with open("./strategyqa/intermediate_data.json", 'r') as f:
    data += json.load(f)

with open("./bamboogle/item_immediate.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))

save_reason_data = []

for item in tqdm(data):
    item_list_exp_data = []
    question = item["question"]
    if "all_reasoning_steps" in item["output"]:
        for idx, reasoning_step in enumerate(item["output"]["all_reasoning_steps"]):
            new_score = reasoning_step[f"Step: {idx}"]["new_score"]
            score_list = reasoning_step[f"Step: {idx}"]["score_list"]
            old_score = max(reasoning_step[f"Step: {idx}"]["score_list"])
            explanation = reasoning_step[f"Step: {idx}"]["explanation"] if "explanation" in reasoning_step[f"Step: {idx}"] else "" 
            reasoning_step_list = reasoning_step[f"Step: {idx}"]["reasoning_step_list"]
            accumulated_output = reasoning_step[f"Step: {idx}"]["accumulated_output"]
            max_idx = score_list.index(max(score_list))
            if explanation!="":
                if new_score>old_score:
                    data_dict = {"question":item["question"], "explanation":explanation, "history_step":"\n".join(accumulated_output), "current_step": reasoning_step_list[max_idx][0], "future_step": "\n".join(reasoning_step_list[max_idx][1:]) if len(reasoning_step_list[max_idx])>1 else "", "orig_score":old_score, "new_score":new_score, "label":1}
                else:
                    data_dict = {"question":item["question"], "explanation":explanation, "history_step":"\n".join(accumulated_output), "current_step": reasoning_step_list[max_idx][0], "future_step": "\n".join(reasoning_step_list[max_idx][1:]) if len(reasoning_step_list[max_idx])>1 else "", "orig_score":old_score, "new_score":new_score, "label":-1}
                format_data = {
                    "messages":
                    [
                    {"role": "user", "content": EXP_SYS_PROMPT+'\n'+EXP_PROMPT.format(question, data_dict["history_step"], data_dict["current_step"], data_dict["future_step"], old_score)},
                    {"role": "assistant", "content":data_dict["explanation"]}
                    ],
                    "label": True if data_dict["label"] == 1 else False,
                    "score": np.exp(data_dict["orig_score"]-data_dict["new_score"]).item()
                }
                item_list_exp_data.append(format_data)
        system_prompt = REASON_PROMPT
            
        user_prompt = f"#\nQuestion: {question}"
        reason_data_list = []
        reason_data_dict_cache = None
        for idx in range(len(item["output"]["reasoning_steps"])):
            if "Search: No" in item["output"]["reasoning_steps"][idx] or "So the final answer is:" in item["output"]["reasoning_steps"][idx]:
                reason_data_dict = {
                        "instruction": system_prompt,
                        "input":  user_prompt+"\n".join(item["output"]["reasoning_steps"][:idx]),
                        "output": item["output"]["reasoning_steps"][idx]
                        }
                reason_data_list.append(reason_data_dict)
                
            else:
                if reason_data_dict_cache==None:
                    part_1 = item["output"]["reasoning_steps"][idx].split('Retrieved documents:', maxsplit=1)[0].strip()
                    reason_data_dict = {
                            "instruction": system_prompt,
                            "input":  user_prompt+"\n".join(item["output"]["reasoning_steps"][:idx]),
                            "output": part_1
                            }
                else:
                    part_1 = item["output"]["reasoning_steps"][idx].split('Retrieved documents:', maxsplit=1)[0].strip()
                    reason_data_dict_cache["output"] += "\n"+part_1
                    reason_data_dict = reason_data_dict_cache
                reason_data_list.append(reason_data_dict)
                part_1 = item["output"]["reasoning_steps"][idx].split('Intermediate answer:', maxsplit=1)[0].strip()
                part_2 = item["output"]["reasoning_steps"][idx].split('Intermediate answer:', maxsplit=1)[1].strip()
                reason_data_dict_cache = {
                        "instruction": system_prompt,
                        "input":  user_prompt+"\n".join(item["output"]["reasoning_steps"][:idx])+'\n'+part_1+'\n'+"Intermediate answer:",
                        "output": part_2
                        }
                        
                
        # 处理 output 的每一行
        for reason_data_dict in reason_data_list:
            if '''The final answer should be "True" or "False"''' in question:
                processed_output = []
                for line in reason_data_dict["output"].split("\n"):
                    # 替换条件
                    if "Yes" in line and "Search: Yes" not in line:
                        line = line.replace("Yes", "True")
                    if "No" in line and "Search: No" not in line:
                        line = line.replace("No", "False")
                    processed_output.append(line)

                # 将处理后的内容重新赋值
                reason_data_dict["output"] = "\n".join(processed_output)
    try:
        pred = item["output"]["pred"]
    except:
        import pdb
        pdb.set_trace()
import pandas as pd
df = pd.DataFrame(item_list_exp_data)

# Adjust column names and split the `messages` column
df['prompt'] = df['messages'].apply(lambda x: x[0]['content'])
df['completion'] = df['messages'].apply(lambda x: x[1]['content'])
df = df.drop(columns=['messages'])
import pdb
pdb.set_trace()    
output_file =  "kto_llama3_exp_refine.parquet"
df.to_parquet(output_file, index=False)