# Adapted from https://github.com/joyheyueya/declarative-math-word-problem

import os
from functools import partial

from codethink._data import TransformedDataset

dir_path = os.path.dirname(os.path.realpath(__file__))

def algebra_input(x):
    return "Question:\n"+x['question']+"\nAnswer:"

def algebra_output(x):
    return x["final_answer"]

# def algebra_fewshot_output(x):
#     return f"Let's think step by step. {x["solution"]} #### {x["final_answer"]}"

def algebra_eval(prediction, ground_truth):
    try:
        prediction = float(prediction)
        ground_truth = float(ground_truth)
        score = 1 if abs(prediction - ground_truth) < 1e-3 else 0
    except Exception as e:
        prediction = str(prediction)
        ground_truth = str(ground_truth)
        score = 1 if prediction == ground_truth else 0

    return score

AlgebraDataset = partial(
    TransformedDataset,
    data_path="csv",
    data_name={
        "test": os.path.join(dir_path, "algebra_test.csv"),
        },
    input_text=algebra_input,
    output_text=algebra_output,
    # fewshot_output_text=algebra_fewshot_output,
    evaluation=algebra_eval,
    test_split="test",
    fewshot_split="test",
)

if __name__ == "__main__":

    dataset = AlgebraDataset(
        num_fewshot=5,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)
