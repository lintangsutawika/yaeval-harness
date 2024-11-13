from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

def lastletterconcat_input(x):
    return "Question:\n"+x["question"].strip()+"\nAnswer:"

def lastletterconcat_output(x):
    return x["answer"]

def lastletterconcat_eval(prediction, ground_truth):
    try:
        score = 1 if (prediction == ground_truth) else 0
    except Exception as e:
        score = 0

    return score

LastLetterConcatDataset = partial(
    TransformedDataset,
    data_path="ChilleD/LastLetterConcat",
    input_text=lastletterconcat_input,
    output_text=lastletterconcat_output,
    eval=lastletterconcat_eval,
    test_split="test",
    fewshot_split="train",
)

if __name__ == "__main__":

    dataset = LastLetterConcatDataset(
        num_fewshot=5,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)