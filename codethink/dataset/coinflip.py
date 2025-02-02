from functools import partial

from codethink._data import TransformedDataset

def coinflip_input(x):
    return "Question:\n"+x["inputs"].strip()+"\nAnswer:"

def coinflip_output(x):
    return x["targets"]

def coinflip_eval(prediction, ground_truth):
    try:
        if prediction == "True":
            prediction = "yes"
        elif prediction == "False":
            prediction = "no"

        score = 1 if (prediction == ground_truth) else 0
    except Exception as e:
        score = 0

    return score

CoinFlipDataset = partial(
    TransformedDataset,
    data_path="skrishna/coin_flip",
    input_text=coinflip_input,
    output_text=coinflip_output,
    evaluation=coinflip_eval,
    test_split="test",
    fewshot_split="train",
)

if __name__ == "__main__":

    dataset = CoinFlipDataset(
        num_fewshot=5,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)
