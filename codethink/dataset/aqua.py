import os
from functools import partial

from codethink.dataset import register_task

from codethink._task import Task, create_task
from codethink._data import TransformedDataset

from codethink.utils import is_runnable_code

dir_path = os.path.dirname(os.path.realpath(__file__))

def aqua_input(x):
    return x["question"]

def aqua_output(x):
    answer_dict = {}
    for _x in x["options"]:
        l, *a = _x.split(")")
        a = ")".join(a)
        answer_dict[l] = a
    return f"{x['correct']} OR {answer_dict[x['correct']]}"

def aqua_eval(prediction, ground_truth):
    score = 0
    letter, number = ground_truth.split(" OR ")
    if prediction == letter:
        score = 1
    elif prediction == number:
        score = 1
    else:
        try:
            if "/" in number:
                gt = eval(number)
            else:
                prediction = float(prediction)
                prediction = match_decimals(prediction, number)
                number = float(number)
                score = 1 if abs(prediction - number) < 1e-3 else 0
        except Exception as e:
            # print(e)
            pass

    return score

AQUADataset = partial(
    TransformedDataset,
    data_path="deepmind/aqua_rat",
    input_text=aqua_input,
    output_text=aqua_output,
    test_split="test",
    fewshot_split="train",
)

def preprocess_routing(x, state):
    current_step = state["current_step"]
    solve_with = state["step"][current_step-1]["output"]
    if "programming language" in solve_with:
        state["system_message"] = "code"
    # elif "natural language" in solve_with:
    else:
        state["system_message"] = "cot"
    return x, state

def postprocess_routing(x, state):
    x = x["response"][0]
    current_step = state["current_step"]
    solve_with = state["step"][current_step-1]["output"]
    if "programming language" in solve_with:
        x = is_runnable_code(x) 
    # elif "natural language" in solve_with:
    else:
        try:
            x = x.split("answer is")[-1].strip()
        except:
            pass
    return x, state

@register_task(
    "aqua_routing_pl_first",
    subtask_list=[
        create_task(
            name="aqua_routing",
            dataset=partial(
                AQUADataset,
                input_text=lambda x: aqua_input(x).replace("\nAnswer:", "")+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="aqua_solve",
            dataset=AQUADataset,
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": aqua_eval},
        ),
    ])
class AquaRouting(Task):
    pass


if __name__ == "__main__":

    dataset = AQUADataset(
        num_fewshot=0,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)
