import re
from functools import partial
from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

@register_task("gsm_hard")
class GSM8KTask(YevalTask):
    data_path="reasoning-machines/gsm-hard"
    input_text=lambda x: "Question: " + x["input"] + "\nAnswer:"
    output_text=lambda x: x["target"]
    test_split="train"
    evaluation={"accuracy": math_eval}
