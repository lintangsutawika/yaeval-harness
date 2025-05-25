import re
from functools import partial
from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

@register_task("aimo_cot")
class AIMOCoT(YevalTask):
    data_path="AI-MO/NuminaMath-CoT"
    input_text=lambda x: x["problem"]
    output_text=lambda x: x["solution"]
    test_split="train"
    evaluation={"accuracy": lambda x, y: -1}
