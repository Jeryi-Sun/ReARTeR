import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import random
import json
import os
import math
import re
import string

# Set your Hugging Face token here
os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_yourkey"

# For reproducibility
set_seed(1234)
random.seed(42)

class Node:
    def __init__(self, question, partial_answer, correct_answer):
        self.question = question
        self.partial_answer = partial_answer
        self.correct_answer = correct_answer
        self.mc_score = None
        self.visits = 0
        self.rollouts = []
        self.visited_rollouts = []

    def add_rollout(self, result):
        self.rollouts.append(result)
        self.visited_rollouts.append(False)

    def increment_visits(self):
        self.visits += 1

    def get_rollouts(self):
        return self.rollouts

def generate_completion(question, partial_answer, tokenizer, model):
    result, bad_gen = model.run_item_mcts(question, "".join(partial_answer))
    return result, bad_gen

def check_correctness(expected_answer, generated_response, metric):
    
    pred = generated_response[-1]

    pred = pred.lower()
    acc_socre = calculate_sub_em(prediction=pred, golden_answers=expected_answer)
    # If expected_answer is a list, check each element
    if isinstance(expected_answer, list):
        for answer in expected_answer:
            answer = answer.lower()
            original_score = metric.token_level_scores(pred, answer)["f1"]
            if original_score > 0.5 or acc_socre:
                return True
    else:
        expected_answer = expected_answer.lower()
        original_score = metric.token_level_scores(pred, expected_answer)["f1"]
        return original_score > 0.5 or acc_socre
    
    return False

def perform_rollouts(node, num_rollouts=5, tokenizer=None, model=None, metric=None):
    correctness_flags = []
    for _ in range(num_rollouts):
        result, bad_gen = generate_completion(node.question, node.partial_answer, tokenizer, model)
        if bad_gen:
            import pdb
            pdb.set_trace()
            print("Bad generation:", result)
            continue
        node.add_rollout(result)
        is_correct = check_correctness(node.correct_answer, result, metric)
        correctness_flags.append(int(is_correct))
    return node.rollouts, correctness_flags

def calculate_mc_score(node, metric):
    correct_count = sum(
        check_correctness(node.correct_answer, r, metric) for r in node.rollouts
    )
    return correct_count / len(node.rollouts) if node.rollouts else 0


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


def select_best_node(nodes, metric):
    best_node = None
    best_rollout_idx = -1
    highest_qu_value = -1
    for node in nodes:
        mc_score = (
            node.mc_score if node.mc_score is not None else calculate_mc_score(node, metric)
        )
        if mc_score in [0, 1]:
            continue
        for idx, rollout in enumerate(node.rollouts):
            if node.visited_rollouts[idx]:
                continue
            q_val = compute_q_value(rollout, mc_score)
            u_val = compute_u_value(node, nodes)
            qu_value = q_val + u_val
            if qu_value > highest_qu_value:
                highest_qu_value = qu_value
                best_node = node
                best_rollout_idx = idx
    if best_rollout_idx != -1 and best_node is not None:
        best_node.visited_rollouts[best_rollout_idx] = True
        return best_node, best_node.rollouts[best_rollout_idx], highest_qu_value
    else:
        return None, None, None


def split_list_middle(lst):
    mid_idx = len(lst) // 2
    part1 = lst[:mid_idx]
    part2 = lst[mid_idx:]
    return part1, part2

def locate_error(node, rollout,tokenizer=None, model=None, metric=None):
    current_span = rollout
    previous_text = [] 
    nodes_to_expand = []
    leaf_nodes = []
    while True:
        if len(current_span) < 2:  
            break
        left_part, right_part = split_list_middle(current_span)
        print("----")
        print(" Left:", left_part)
        print(" Right:", right_part)
        new_node = Node(
            node.question, previous_text + left_part, node.correct_answer
        )
        perform_rollouts(new_node, tokenizer=tokenizer, model=model, metric=metric)
        mc_score = calculate_mc_score(new_node, metric)
        new_node.mc_score = mc_score
        if mc_score == 1:
            break
        elif mc_score > 0:
            current_span = right_part
            previous_text += left_part
            nodes_to_expand.append(new_node)
        else:
            current_span = left_part
            leaf_nodes.append(new_node)
    print("----")
    return nodes_to_expand, leaf_nodes
def compute_q_value(rollout_text, mc_score, alpha=0.5, beta=0.9, max_length=6):
    part1 = alpha ** (1 - mc_score)
    part2 = beta ** (len(rollout_text) / max_length)
    return part1 * part2

def compute_u_value(node, all_nodes, exploration_param=0.125):
    total_visits = sum(n.visits for n in all_nodes)
    numerator = math.sqrt(total_visits)
    denominator = 1 + node.visits
    return exploration_param * (numerator / denominator)

def append_to_json(filename, data_entry):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
    else:
        data = []
    data.append(data_entry)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data appended to {filename}")

def process_annotations(question, answer, nodes, filename='nodes_data.json', tokenizer=None, model=None, metric=None):
    print("++++++")
    iteration = 0
    leaf_nodes = []
    try:
        while True:
            node, rollout, max_qu = select_best_node(nodes, metric)
            if node is not None and node.partial_answer != []:
                new_entry = {
                    "question": question,
                    "partial_answer": node.partial_answer,
                    "mc_score": node.mc_score,
                    "type": "best"
                }
                append_to_json(filename, new_entry)
                iteration += 1
                if iteration > 20:
                    break
            if node is None:
                break
            print()
            print("[Selected Node]")
            print(node)
            print("  Rollout:", rollout, " || QU Value:", max_qu)
            node.increment_visits()
            expanded_nodes, leaves = locate_error(node, rollout, tokenizer=tokenizer, model=model, metric=metric)
            if not expanded_nodes:
                continue
            nodes.extend(
                n for n in expanded_nodes if n is not None and n.partial_answer != []
            )
            leaf_nodes.extend(leaves)
        for leaf_node in leaf_nodes:
            new_entry = {
                "question": question,
                "partial_answer": node.partial_answer,
                "mc_score": leaf_node.mc_score,
                "type": "leaf"
            }
            append_to_json(filename, new_entry)
        for node in nodes:
            if node is not None and node.partial_answer != []:
                new_entry = {
                    "question": question,
                    "partial_answer": node.partial_answer,
                    "mc_score": node.mc_score,
                    "type": "add"
                }
                append_to_json(filename, new_entry)
        print("++++++")
    except Exception as e:
        print("Error occurred during processing:", e)  
        
        for leaf_node in leaf_nodes:
            new_entry = {
                "question": question,
                "partial_answer": leaf_node.partial_answer,
                "mc_score": leaf_node.mc_score,
                "type": "leaf"
            }
            append_to_json(filename, new_entry)
        
        for node in nodes:
            if node is not None and node.partial_answer != []:
                new_entry = {
                    "question": question,
                    "partial_answer": node.partial_answer,
                    "mc_score": node.mc_score,
                    "type": "add"
                }
                append_to_json(filename, new_entry)
        
        print("++++++")  
