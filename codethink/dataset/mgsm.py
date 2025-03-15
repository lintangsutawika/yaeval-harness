import re
from functools import partial

from codethink.dataset import register_task

from codethink._task import Task, create_task
from codethink._data import TransformedDataset

from codethink.utils import is_runnable_code

from codethink.dataset.utils import remove_boxed, last_boxed_only_string

from math_verify import parse, verify

def mgsm_input(x):
    return x['question']

def mgsm_output(x):
    return x['answer_number']

def mgsm_eval(prediction, ground_truth):
    try:
        prediction = parse(prediction)
        ground_truth = parse(ground_truth)
        return int(verify(ground_truth, prediction))
    except Exception as e:
        print(f"Error: {e}")
        pass

    return 0

MGSMDataset = partial(
    TransformedDataset,
    data_path="juletxara/mgsm",
    input_text=mgsm_input,
    output_text=mgsm_output,
    test_split="test",
)

def postprocess(x, state):
    x = x["response"][0]
    return remove_boxed(last_boxed_only_string(x)), state

@register_task(
    "mgsm_en_raw",
    dataset=partial(MGSMDataset, data_name="en"),
    postprocessor=lambda x, state: (x["response"][0], state),
    evaluation={"score": lambda x, y: -1},
    )
class MGSM_EN_RAW_Task(Task):
    pass

@register_task(
    "mgsm_en",
    dataset=partial(MGSMDataset, data_name="en"),
    postprocessor=postprocess,
    evaluation={"score": mgsm_eval},
    )
class MGSM_EN_Task(Task):
    pass

@register_task(
    "mgsm_ja",
    dataset=partial(MGSMDataset, data_name="ja"),
    postprocessor=postprocess,
    evaluation={"score": mgsm_eval},
    )
class MGSM_JP_Task(Task):
    pass
