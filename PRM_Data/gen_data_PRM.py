import json
import logging
from datetime import datetime
from module import Node, perform_rollouts, process_annotations, calculate_mc_score
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from flashrag.pipeline import ReasoningPipeline
from flashrag.config import Config
import argparse
from flashrag.evaluator.metrics import F1_Score

def load_json_file(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        list: A list of dictionaries containing the problem and final answer.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)[:200]
    return data

def setup_logging(log_file):
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_generation_model(args):
    save_note = "reasoning"
    config_dict = {"save_note": save_note, "gpu_id": args.gpu_id, "dataset_name": args.dataset_name}

    # preparation
    config = Config("my_config.yaml", config_dict)

    pipeline = ReasoningPipeline(config)
    F1_s = F1_Score(config)
    return pipeline, F1_s

def main():
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--method_name", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--gpu_id", type=str)

    args = parser.parse_args()
    pipline_model, F1_s = get_generation_model(args)
    # Path to the JSON file and log file
    json_file_path = f'./{args.dataset_name}/{args.dataset_name}_mcts_data.json'
    # json_file_path = f'./{args.dataset_name}/{args.dataset_name}_data_mc.json' # for difficult data
    current_time = datetime.now().strftime("%m%d%H%M")
    #log_file_path = f'./{args.dataset_name}/logs/processing_log_{current_time}.log'
    log_file_path = f'./{args.dataset_name}/logs/processing_log_{current_time}.log'

    
    # Set up logging
    setup_logging(log_file_path)
    
    # Start the process and log it
    logging.info("Started processing the JSON file.")
    
    # Load the JSON data
    data = load_json_file(json_file_path)
    
    # Process each problem and its final answer
    count_hard_data = 0
    for i, item in enumerate(data):
        if item["mc_score"]!=0.0:
            print(item["mc_score"])
            continue
        count_hard_data+=1
        if count_hard_data>20:
            break
        problem = item.get('problem', 'No problem found')
        if args.dataset_name=="strategyqa":
            problem += "The final answer should be 'True' or 'False'."
        final_answer = item.get('final_answer', 'No answer found')
        
        # Print to console
        print(f"Problem {i + 1}: {problem}")
        print(f"Final Answer: {final_answer}")
        
        # Log each problem and answer
        logging.info(f"Processed Problem {i + 1}: {problem}")
        logging.info(f"Final Answer: {final_answer}")
        
        # Initialize the root node and perform rollouts
        nodes = []
        root_node = Node(problem, [], final_answer)
        max_rollouts = 5
        rollouts, correctness_flags = perform_rollouts(root_node, max_rollouts, model=pipline_model, metric=F1_s)
    

        mc_score = calculate_mc_score(root_node, metric=F1_s)
        root_node.mc_score = mc_score

        nodes.append(root_node)
        print(f"correctness_flags: {correctness_flags}")
        print(f"mc_score: {mc_score}")
        logging.info(f"correctness_flags: {correctness_flags}")
        logging.info(f"mc_score: {mc_score}")

        # Check if further processing is needed
        if 0 < sum(correctness_flags) < max_rollouts:
            print("Processing annotations ...\n")
            filename = f"./{args.dataset_name}/results/{i+1}_nodes_data.json"
            process_annotations(problem, final_answer, nodes, filename, model=pipline_model, metric=F1_s)
        
    # Log completion
    logging.info("Finished processing the JSON file.")

if __name__ == "__main__":
    main()
