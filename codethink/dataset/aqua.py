import os
from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

dir_path = os.path.dirname(os.path.realpath(__file__))

def aqua_input(x):
    return x["question"]

def aqua_output(x):
    answer_dict = {}
    for _x in x["options"]:
        l, *a = _x.split(")")
        a = ")".join(a)
        answer_dict[l] = a
    return x["correct"], answer_dict[x["correct"]]

def aqua_fewshot_output(x):
    return f"Let's think step by step. {x['rationale']} #### {x['correct']}"

def aqua_eval(prediction, ground_truth):

    score = 0
    letter, number = ground_truth.split(" OR ")
    if prediction == letter:
        score = 1
    elif prediction == number:
        score = 1
    elif prediction == full_answer:
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
    evaluation=aqua_eval,
    fewshot_output_text=aqua_fewshot_output,
    test_split="test",
    fewshot_split="train",
)

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
