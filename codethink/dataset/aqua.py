from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

def aqua_input(x):
    return "Question:\n"+x["question"]+"\n".join(x["options"])+"\nAnswer:"

def aqua_output(x):
    return x["Answer"]

def aqua_fewshot_output(x):
    return f"Let's think step by step. {x["rationale"]} #### {x["correct"]}"

AQUADataset = partial(
    TransformedDataset,
    data_path="deepmind/aqua_rat",
    input_text=aqua_input,
    output_text=aqua_output,
    fewshot_output_text=aqua_fewshot_output,
    test_split="test",
    fewshot_split="train",
)

if __name__ == "__main__":

    dataset = AQUADataset(
        num_fewshot=5,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)