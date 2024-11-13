import os
from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

dir_path = os.path.dirname(os.path.realpath(__file__))

def tabmwp_input(x):
    return "Table:\n"+x["table"]+"\nQuestion:\n"+x["question"].strip()+"\nAnswer:"

def tabmwp_output(x):
    return x["answer"]

def tabmwp_fewshot_output(x):
    return f"Let's think step by step. {x["solution"]} #### {x["answer"]}"

def tabmwp_eval(prediction, ground_truth):
    try:
        prediction = float(prediction)
        ground_truth = float(ground_truth)
        score = 1 if abs(prediction - ground_truth) < 1e-3 else 0
    except Exception as e:
        prediction = str(prediction)
        ground_truth = str(ground_truth)
        score = 1 if prediction == ground_truth else 0

    return score

TabMWPDataset = partial(
    TransformedDataset,
    data_path="json",
    data_name={
        "test": os.path.join(dir_path, "tabmwp_test.jsonl"),
        "dev": os.path.join(dir_path, "tabmwp_dev.jsonl"),
        },
    input_text=tabmwp_input,
    output_text=tabmwp_output,
    fewshot_output_text=tabmwp_fewshot_output,
    eval=tabmwp_eval,
    test_split="test",
    fewshot_split="dev",
)

if __name__ == "__main__":

    dataset = TabMWPDataset(
        num_fewshot=5,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)