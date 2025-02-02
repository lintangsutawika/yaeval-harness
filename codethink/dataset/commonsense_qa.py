import re
import os
from functools import partial

from codethink.dataset import register_task

from codethink._task import Task
from codethink._data import TransformedDataset

dir_path = os.path.dirname(os.path.realpath(__file__))

def cqa_input(x):
    choices = x["choices"]
    choice_list = [f"{a}. {b}" for a,b in list(zip(choices["label"], choices["text"]))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either A, B, C, D or E.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def cqa_output(x):
    return x["answerKey"]

def cqa_eval(prediction, ground_truth):
    score = 0
    try:
        prediction = prediction.split(".")[0]
        if prediction in ["A", "B", "C", "D", "E"]:
            if prediction == ground_truth:
                score = 1
    except Exception as e:
        pass
    return score

CommonsenseQADataset = partial(
    TransformedDataset,
    data_path="tau/commonsense_qa",
    input_text=cqa_input,
    output_text=cqa_output,
    evaluation=cqa_eval,
    test_split="validation",
)

CommonsenseQARoutingDataset = partial(
    TransformedDataset,
    data_path="tau/commonsense_qa",
    input_text=lambda x: x['question']+"\n\nWhich method is the best way to solve this problem?",
    output_text=lambda x: "natural language",
    test_split="validation",
)

@register_task(
    "commonsense_qa_routing",
    dataset=CommonsenseQARoutingDataset,
    postprocessor=lambda x, state: x["response"][0].split("\n\n")[0].strip(),
    evaluation=lambda x, y: 1 if re.sub(r'[^\w\s]', '', x.lower()) == re.sub(r'[^\w\s]', '', y.lower()) else 0,
    )
class CommonsenseQARoutingTask(Task):
    pass


if __name__ == "__main__":

    dataset = CommonsenseQADataset(
        num_fewshot=0,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)
