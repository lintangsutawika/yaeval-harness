import re
from functools import partial

from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage
from yeval.metrics import math_eval

@register_task("AIME2025-I")
class AIME2025ITask(YevalTask):
    data_path="opencompass/AIME2025"
    data_name="AIME2025-I"
    input_text=lambda x: x["question"]
    output_text=lambda x: x["answer"]
    test_split="test"
    evaluation={"accuracy": math_eval}
    logging=log_token_usage

@register_task("AIME2025-II")
class AIME2025IITask(AIME2025ITask):
    data_name="AIME2025-II"

@register_task("AIME2024")
class AIME2025Task(YevalTask):
    data_path="HuggingFaceH4/aime_2024"
    input_text=lambda x: x["problem"]
    output_text=lambda x: x["answer"]
    test_split="train"
    evaluation={"accuracy": math_eval}
    logging=log_token_usage

