# score_list 中计算 original score

import json
from flashrag.prompt.reasoning_examplars import REASON_PROMPT, EXP_PROMPT, CORRECT_PROMPT, EXP_SYS_PROMPT, SUMMATY_PROMPT, SUMMATY_PROMPT_STR
from tqdm import tqdm
import re
import string
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_dataset_answer(data):
    if any(choice == [] for choice in data.choices):
        golden_answers_list = data.golden_answers
    else:
        # multi-choice dataset
        all_choices_list = data.choices
        golden_choice_idx_list = data.golden_answers
        golden_answers_list = [
            [choices[idx] for idx in idx_list]
            for choices, idx_list in zip(all_choices_list, golden_choice_idx_list)
        ]

    return golden_answers_list


def calculate_sub_em(prediction: str, golden_answers: list) -> float:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1.0
            break
    return score


data = []
with open("/home/zangxiaoxue/sunzhongxiang/flashrag_data/retrieval-corpus/output/musique_2024_12_09_16_36_eval/intermediate_data.json", 'r') as f:
    data += json.load(f)

with open("/home/zangxiaoxue/sunzhongxiang/flashrag_data/retrieval-corpus/output/hotpotqa_2024_12_04_07_41_reasoning/intermediate_data.json", 'r') as f:
    data += json.load(f)


with open("/home/zangxiaoxue/sunzhongxiang/flashrag_data/retrieval-corpus/output/2wikimultihopqa_2024_12_04_23_11_reasoning/intermediate_data.json", 'r') as f:
    data += json.load(f)

with open("/home/zangxiaoxue/sunzhongxiang/flashrag_data/retrieval-corpus/output/strategyqa_2024_12_03_17_46_reasoning/intermediate_data.json", 'r') as f:
    data += json.load(f)

with open("/home/zangxiaoxue/sunzhongxiang/flashrag_data/retrieval-corpus/output/bamboogle_2024_12_06_00_30_reasoning/intermediate_data.json", 'r') as f:
    data += json.load(f)


save_summary_data = []

        # messages = [
        #     [{
        #     "role": "user",
        #     "content":  self.SUM.format(question, "\n".join(reasoning_steps))
        #     }]
        #     ]

for item in tqdm(data):
    question = item["question"]
    if "all_reasoning_steps" in item["output"]:
        user_prompt = f"#\nQuestion: {question}"
        reason_data_list = []
        reason_data_dict_cache = None
        reasoning_steps = item["output"]["reasoning_steps"]
        pred = item["output"]["pred"]
        reason_data_dict = {
                    "instruction": "",
                    "input":  SUMMATY_PROMPT_STR.format(question, "\n".join(reasoning_steps)),
                    "output": pred
                    }
        golden_answers = item["golden_answers"]
        score = calculate_sub_em(prediction=pred, golden_answers=golden_answers)
        if score:
            save_summary_data.append(reason_data_dict)


                        
import pdb
pdb.set_trace()
# 整理一下格式，都 save 到 llama-factory 下面
with open("/home/zangxiaoxue/sunzhongxiang/Rag_Reasoning/LLaMA-Factory-2/LLaMA-Factory/data/sft_llama3_summary.json", 'w') as f:
    json.dump(save_summary_data, f, ensure_ascii=False, indent=4)        
# with open("/home/zangxiaoxue/sunzhongxiang/Rag_Reasoning/LLaMA-Factory-2/LLaMA-Factory/data/kto_llama3_exp_refine.json", 'w') as f:
#     json.dump(save_data_exp, f, ensure_ascii=False, indent=4)      
