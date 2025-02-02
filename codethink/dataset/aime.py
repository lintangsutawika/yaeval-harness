from functools import partial

from codethink._data import TransformedDataset

def aime_input(x):
    return "Question:\n"+x["Question"].strip()+"\nAnswer:"

def aime_output(x):
    return x["Answer"]

def aime_eval(prediction, ground_truth):
    try:
        score = 1 if (float(prediction) == float(ground_truth)) else 0
    except Exception as e:
        score = 0

    return score

AIMEDataset = partial(
    TransformedDataset,
    data_path="qq8933/AIME_1983_2024",
    input_text=aime_input,
    output_text=aime_output,
    evaluation=aime_eval,
    test_split="train",
)

if __name__ == "__main__":

    dataset = AIMEDataset(
        num_fewshot=0,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)
