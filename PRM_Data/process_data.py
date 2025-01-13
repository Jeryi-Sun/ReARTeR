import json
data = []
with open("/share/sunzhongxiang/flashrag_data/retrieval-corpus/dataset/FlashRAG_datasets/hotpotqa/train.jsonl",'r') as f:
    for line in f:
        data.append(json.loads(line))
types = {}
mcts_data = []

for d in data:
    if d['metadata']["level"] in types:
        types[d['metadata']["level"]] += 1
    else:
        types[d['metadata']["level"]] = 1
    if d['metadata']["level"] == "hard":
        mcts_data.append({"problem":d["question"], "final_answer":d["golden_answers"][0]})
print(len(mcts_data))
with open("./hotpotqa/hotpotqa_mcts_data.json",'w') as f:
    json.dump(mcts_data,f, ensure_ascii=False)

import json
data = []
with open("/share/sunzhongxiang/flashrag_data/retrieval-corpus/dataset/FlashRAG_datasets/2wikimultihopqa/train.jsonl",'r') as f:
    for line in f:
        data.append(json.loads(line))
types = {"compositional":0,"comparison":0,"inference":0,"bridge_comparison":0}
limit_num = {"compositional":160,"comparison":120,"inference":40,"bridge_comparison":80}
mcts_data = []


for d in data:
    if types[d['metadata']["type"]] < limit_num[d['metadata']["type"]]:
        mcts_data.append({"problem":d["question"], "final_answer":d["golden_answers"]})
        types[d['metadata']["type"]] += 1
        

print(len(mcts_data))
with open("./2wikimultihopqa/2wikimultihopqa_mcts_data.json",'w') as f:
    json.dump(mcts_data,f, ensure_ascii=False)
    
import json
data = []
with open("/share/sunzhongxiang/flashrag_data/retrieval-corpus/dataset/FlashRAG_datasets/musique/train.jsonl",'r') as f:
    for line in f:
        data.append(json.loads(line))
mcts_data = []


for d in data:
    if d['metadata']["answerable"]==True:
        mcts_data.append({"problem":d["question"], "final_answer":d["golden_answers"]})
        

print(len(mcts_data))
with open("./musique/musique_mcts_data.json",'w') as f:
    json.dump(mcts_data,f, ensure_ascii=False)
        
# import json
# data = []
# with open("/share/sunzhongxiang/flashrag_data/retrieval-corpus/dataset/FlashRAG_datasets/strategyqa/train.jsonl",'r') as f:
#     for line in f:
#         data.append(json.loads(line))
# mcts_data = []


# for d in data[1000:]:
#     mcts_data.append({"problem":d["question"], "final_answer":[str(ans) for ans in d["golden_answers"]]})
        

# print(len(mcts_data))
# with open("./strategyqa/strategyqa_mcts_data.json",'w') as f:
#     json.dump(mcts_data,f, ensure_ascii=False)