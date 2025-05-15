import re
from functools import partial
from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

def gsm_input(x):
    return f"Question:\n{x['question']}\nAnswer:"

def gsm_output(x):
    answer = x["solution"]
    answer = answer.split("####")[-1].strip()
    return answer

@register_task("gsm_plus")
class GSMPlusTask(YevalTask):
    data_path="qintongli/GSM-Plus"
    input_text=gsm_input
    output_text=gsm_output
    test_split="test"
    evaluation={"accuracy": math_eval}
