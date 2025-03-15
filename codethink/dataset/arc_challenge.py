import os
from functools import partial

from codethink.dataset import register_task
from codethink._task import Task
from codethink._data import TransformedDataset
from codethink.dataset.utils import get_boxed_answer, math_eval


dir_path = os.path.dirname(os.path.realpath(__file__))

def arc_input(x):
    choices = x["choices"]
    choice_list = [f"{a}. {b}" for a,b in list(zip(choices["label"], choices["text"]))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either A, B, C, or D.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def arc_output(x):
    return x["answerKey"]

def arc_eval(prediction, ground_truth):
    score = 0
    try:
        prediction = prediction.split(".")[0]
        if prediction in ["A", "B", "C", "D"]:
            if prediction == ground_truth:
                score = 1
    except Exception as e:
        pass
    return score

ARCDataset = partial(
    TransformedDataset,
    data_path="allenai/ai2_arc",
    data_name="ARC-Challenge",
    input_text=arc_input,
    output_text=arc_output,
    # fewshot_output_text=arc_fewshot_output,
    # evaluation=arc_eval,
    test_split="test",
)

@register_task(
    "arc_challenge_boxed",
    dataset=ARCDataset,
    postprocessor=get_boxed_answer,
    evaluation={"accuracy": math_eval},
    )
class ARCChallengeBoxed(Task):
    pass


if __name__ == "__main__":

    dataset = ARCDataset(
        num_fewshot=0,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)
