from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

def multiarith_input(x):
    return "Question:\n"+x["question"].strip()+"\nAnswer:"

def multiarith_output(x):
    return x["final_ans"]

# def multiarith_fewshot_output(x):
#     return f"Let's think step by step. {x["rationale"]} #### {x["final_ans"]}"

def multiarith_eval(prediction, ground_truth):
    try:
        prediction = str(prediction).replace(",", "")
        prediction = int(prediction)
        ground_truth = int(ground_truth)
        score = 1 if abs(prediction - ground_truth) < 1e-3 else 0
    except Exception as e:
        score = 0

    return score

MultiArithDataset = partial(
    TransformedDataset,
    data_path="ChilleD/MultiArith",
    input_text=multiarith_input,
    output_text=multiarith_output,
    # fewshot_output_text=multiarith_fewshot_output,
    eval=multiarith_eval,
    test_split="test",
    fewshot_split="train",
)

if __name__ == "__main__":

    dataset = MultiArithDataset(
        num_fewshot=5,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)