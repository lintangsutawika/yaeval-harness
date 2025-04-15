import re
from functools import partial
from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

def gsm_input(x):
    return f"Question:\n{x['question']}\nAnswer:"

def gsm_output(x):
    answer = x["answer"]
    answer = answer.split("####")[-1].strip()
    answer = re.findall(r'\d+', answer)[0]
    return answer

@register_task("gsm_symbolic")
class GSMSymbolicTask(YevalTask):
    data_path="apple/GSM-Symbolic"
    data_name="main"
    input_text=gsm_input
    output_text=gsm_output
    test_split="test"
    evaluation={"accuracy": math_eval}

@register_task("gsm_symbolic_p1")
class GSMSymbolicP1Task(GSMSymbolicTask):
    data_name="p1"

@register_task("gsm_symbolic_p2")
class GSMSymbolicP1Task(GSMSymbolicTask):
    data_name="p2"