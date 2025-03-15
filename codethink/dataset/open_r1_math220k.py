import re
from functools import partial

from codethink.dataset import register_task

from codethink._task import Task, create_task
from codethink._data import TransformedDataset

from codethink.utils import is_runnable_code
from codethink.dataset.utils import remove_boxed, last_boxed_only_string

def math_220k_input(x):
    return x['problem']

def math_220k_output(x):
    return x["answer"]

def shuffle_dataset(dataset):
    dataset = dataset.shuffle()
    return dataset
    return dataset.flatten_indices()

OpenR1Math220KDataset = partial(
    TransformedDataset,
    data_path="open-r1/OpenR1-Math-220k",
    data_name="default",
    input_text=math_220k_input,
    output_text=math_220k_output,
    # preprocessing=shuffle_dataset,
    test_split="train",
)

@register_task(
    "open_r1_math_220k",
    dataset=OpenR1Math220KDataset,
    evaluation={"score": lambda x,y: -1},
    )
class OpenR1Math220KTask(Task):
    pass

@register_task(
    "open_r1_math_220k_reasoning",
    dataset=partial(
        OpenR1Math220KDataset,
        input_text=lambda x: x["solution"]
        ),
    postprocessor=lambda x,y:(x["response"][0], y), 
    evaluation={"score": lambda x,y: -1},
    )
class OpenR1220KReasoningTraces(Task):
    pass

if __name__ == "__main__":
    pass
