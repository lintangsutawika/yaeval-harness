import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

MODEL = "meta-llama-Llama-3.1-8B-Instruct"
DATA_PATH="/home/lsutawik/01-code-reasoning/data/output-preference-responses/"
cot_code = load_dataset("json", data_files=f"{DATA_PATH}{MODEL}-mathinstruct_cot-code-reponses/output.jsonl")['train']
cot_none = load_dataset("json", data_files=f"{DATA_PATH}{MODEL}-mathinstruct_cot-none-reponses/output.jsonl")['train']
pot_cot  = load_dataset("json", data_files=f"{DATA_PATH}{MODEL}-mathinstruct_pot-cot-responses/output.jsonl")['train']
pot_none = load_dataset("json", data_files=f"{DATA_PATH}{MODEL}-mathinstruct_pot-none-responses/output.jsonl")['train']

data = {
    'task':[],
    'solution':[],
    'question':[],
    'response_i':[],
    'response_j':[]
    }

d = load_dataset("TIGER-Lab/MathInstruct")['train']
cot = d.filter(lambda example: "CoT" in example["source"])
pot = d.filter(lambda example: "PoT" in example["source"])

# COT Samples
for x, x_code, x_none in tqdm(zip(cot, cot_code, cot_none), total=len(cot)):
    task = x['source']
    solution = "CoT" if "CoT" in task else "PoT"
    question = x_code['user_input']
    score = x_code['score']
    if score:
        solution = "PoT"
        chosen = x_code['system_output'][0]
        rejected = x_none['system_output'][0]
    else:
        solution = "CoT"
        chosen = x['output']
        rejected = x_none['system_output'][0]
    data['task'].append(task)
    data['solution'].append(solution)
    data['question'].append(question)
    data['response_i'].append(chosen)
    data['response_j'].append(rejected)

# POT Samples
for x, x_cot, x_none in tqdm(zip(pot, pot_cot, pot_none), total=len(pot)):
    task = x['source']
    solution = "CoT" if "CoT" in task else "PoT"
    question = x_cot['user_input']
    score = x_cot['score']

    rejected = x_none['system_output'][0]
    chosen = "```"+x_none['ground_truth']+"```"
    
    data['task'].append(task)
    data['solution'].append(solution)
    data['question'].append(question)
    data['response_i'].append(chosen)
    data['response_j'].append(rejected)

df = pd.DataFrame(data=data)
df.to_csv("mathinstruct.csv", index=False, escapechar='\\')
df = pd.read_csv("mathinstruct.csv")
df = df[~df['question'].isnull()]
print(len(df))
df = df[~df['response_i'].isnull()]
print(len(df))
df = df[~df['response_j'].isnull()]
print(len(df))
df.to_csv("mathinstruct.csv", index=False, escapechar='\\')
df = pd.read_csv("mathinstruct.csv")
cot = df[df['solution'] == 'CoT'].reset_index()
pot = df[df['solution'] == 'PoT'].reset_index()

for p in [100, 75, 50, 25]:
    _max = 70000
    pidx = p/100*_max
    cidx = _max - pidx
    pot_df = pot.loc[:pidx]
    cot_df = cot.loc[:cidx]
    _df = pd.concat([pot_df, cot_df], axis=0)
    _df.to_csv(f"mathinstruct_PL_{p}p.csv", index=False)

    
