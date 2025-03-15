import re
import os
from functools import partial

from codethink.dataset import register_task

from codethink._task import Task
from codethink._data import TransformedDataset

from codethink.utils import is_runnable_code

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
    test_split="validation",
)

CommonsenseQARoutingDataset = partial(
    TransformedDataset,
    data_path="tau/commonsense_qa",
    input_text=lambda x: x['question']+"\n\nWhich method is the best way to solve this problem?",
    output_text=lambda x: "natural language",
    test_split="validation",
)

def postprocess_PL_or_NL(x, state):
    x = x["response"][0]
    exec_result = is_runnable_code(x) 
    if exec_result:
        return exec_result, state
    else:
        try:
            x = x.split("answer is")[-1].strip()
        except:
            pass
    return x, state

@register_task(
    "commonsense_qa",
    dataset=CommonsenseQADataset,
    postprocessor=postprocess_PL_or_NL,
    evaluation={"accuracy": cqa_eval},
    )
class CommonsenseQATask(Task):
    pass

def match_routing(prediction, ground_truth):
    prediction = prediction.strip().lower()
    ground_truth = ground_truth.strip().lower()
    if re.sub(r'[^\w\s]', '', prediction) == re.sub(r'[^\w\s]', '', ground_truth):
        return 1
    elif ground_truth in prediction:
        return 1
    return 0

@register_task(
    "routing_commonsense_qa",
    dataset=CommonsenseQARoutingDataset,
    postprocessor=lambda x, state: (x["response"][0], state),
    sampling_args={"stop": ["\n\n", "\n"]},
    evaluation={"accuracy": match_routing},
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
