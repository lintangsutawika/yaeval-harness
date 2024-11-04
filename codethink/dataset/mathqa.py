from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

def mathqa_input(x):
    return f"Question:\n{x["Problem"]}\n{x["options"]}\nAnswer:"

def mathqa_output(x):
    return x["correct"]

def mathqa_fewshot_output(x):
    return f"Let's think step by step. {x["Rationale"]} #### {x["correct"]}"

MathQADataset = partial(
    TransformedDataset,
    data_path="allenai/math_qa",
    input_text=mathqa_input,
    output_text=mathqa_output,
    fewshot_output_text=mathqa_fewshot_output,
    test_split="test",
    fewshot_split="train",
)

if __name__ == "__main__":

    dataset = MathQADataset(
        num_fewshot=5,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)