import os
from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

dir_path = os.path.dirname(os.path.realpath(__file__))

def finqa_input(x):

    pre_text = "\n".join(x['pre_text'])
    post_text = "\n".join(x['post_text'])
    table = "\n".join([" | ".join(line) for line in x['table_ori']])
    question = x['qa']['question']

    return f"{pre_text}\n\nTable:\n{table}\n\n{post_text}\n{question}\nAnswer"

def finqa_output(x):
    return x['qa']['answer']

# def finqa_fewshot_output(x):
#     return f"Let's think step by step. {x["solution"]} #### {x["answer"]}"

def finqa_eval(prediction, ground_truth):
    try:
        prediction = float(prediction)
        ground_truth = float(ground_truth)
        score = 1 if abs(prediction - ground_truth) < 1e-3 else 0
    except Exception as e:
        prediction = str(prediction)
        ground_truth = str(ground_truth)
        score = 1 if prediction == ground_truth else 0

    return score

FinQADataset = partial(
    TransformedDataset,
    data_path="json",
    data_name={
        "test": os.path.join(dir_path, "finqa_test.jsonl"),
        },
    input_text=finqa_input,
    output_text=finqa_output,
    # fewshot_output_text=finqa_fewshot_output,
    eval=finqa_eval,
    test_split="test",
)

if __name__ == "__main__":

    dataset = FinQADataset(
        num_fewshot=0,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)