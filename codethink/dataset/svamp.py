from functools import partial

from codethink.dataset.data import TransformedDataset
# from data import TransformedDataset

def svamp_input(x):
    return "Question:\n"+x["question_concat"]+"\nAnswer:"

def svamp_output(x):
    return x["Answer"]

def svamp_fewshot_output(x):
    return f"Let's think step by step, this is {x["Type"].lower()} problem. So we could write this as {x["Equation"]}. #### {x["Answer"]}"

SVAMPDataset = partial(
    TransformedDataset,
    data_path="ChilleD/SVAMP",
    input_text=svamp_input,
    output_text=svamp_output,
    fewshot_output_text=svamp_fewshot_output,
    test_split="test",
    fewshot_split="train",
)

if __name__ == "__main__":

    dataset = SVAMPDataset(
        num_fewshot=5,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)