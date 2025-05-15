import os

from yeval.task import register_task, YevalTask
from yeval.task.ifeval.evaluation_lib import (
    test_instruction_following_strict,
    test_instruction_following_loose
    )

path = os.path.dirname(__file__)

@register_task("ifeval")
class IFEvalLikeTask(YevalTask):
    data_path="google/IFEval"
    input_text=lambda x: x['prompt']
    output_text=lambda x: {
        "instruction_id_list": x["instruction_id_list"],
                     "kwargs": x["kwargs"],
        }
    test_split="train"
    evaluation={
        "strict": test_instruction_following_strict,
        "loose": test_instruction_following_loose,
        }
