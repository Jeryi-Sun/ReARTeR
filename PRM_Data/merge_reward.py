import json
from pathlib import Path

# File paths
files_to_merge = [
    "/home/zangxiaoxue/sunzhongxiang/Rag_Reasoning/openr/data/musique/rft_results/reward_data/rft_data.json",
    "/home/zangxiaoxue/sunzhongxiang/Rag_Reasoning/openr/data/hotpotqa/rft_results/reward_data/rft_data.json",
    "/home/zangxiaoxue/sunzhongxiang/Rag_Reasoning/openr/data/2wikimultihopqa/rft_results/reward_data/rft_data.json",
    "/home/zangxiaoxue/sunzhongxiang/Rag_Reasoning/openr/data/strategyqa/rft_results/reward_data/rft_data.json"
]

# Output file
output_file = "/home/zangxiaoxue/sunzhongxiang/Rag_Reasoning/LLaMA-Factory-2/LLaMA-Factory/data/rft_data.json"

# Load and merge data
merged_data = []

for file_path in files_to_merge:
    with open(file_path, 'r') as f:
        data = json.load(f)
        merged_data.extend(data)

# Save merged data
with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)
