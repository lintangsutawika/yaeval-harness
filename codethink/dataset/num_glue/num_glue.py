import os
from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

dir_path = os.path.dirname(os.path.realpath(__file__))

def num_glue_input(x):
    if x['type'] == "Type_7":
        return "Statement1:\n"+x['statement1']+"\nStatement2:\n"+x['statement2']+"\nQuestion:\n"+x['options'].strip()+"\nAnswer:"
    elif (x['type'] == "Type_6") or (x['type'] == "Type_5"):
        return "Passage:\n"+x['passage']+"\nQuestion:\n"+x['question']+"\nAnswer:"
    elif x["type"] == "Type_4":
        return "Question:\n"+x['question']+"\nFill in the blank.\nAnswer:"
    elif x["type"] == "Type_3":
        question = x["question"][:-1]
        return "Question:\n"+question+f"\n{x['Option1']} or {x['Option2']}?"
    else:
        return "Question:\n"+x['question']+"\nAnswer:"

def num_glue_output(x):
    if x["type"] == "Type_6":
        return x["answer"]["spans"][0]
    elif x["type"] == "Type_5":
        return x["answer"]["number"]
    elif x["type"] == "Type_3":
        return x[x["answer"].replace(" ", "")]
    else:
        return x["answer"]

def num_glue_eval(prediction, ground_truth):
    try:
        prediction = float(prediction)
        ground_truth = float(ground_truth)
        score = 1 if abs(prediction - ground_truth) < 1e-3 else 0
    except Exception as e:
        prediction = str(prediction).lower()
        ground_truth = str(ground_truth).lower()
        score = 1 if prediction == ground_truth else 0
        if score == 0:
            if prediction.startswith(ground_truth):
                score = 1
            elif prediction.endswith(ground_truth):
                score = 1
            elif prediction.startswith("the"):
                prediction = prediction[3:].strip()
                score = 1 if prediction == ground_truth else 0

    return score

num_glue_tasks = {}
for num in range(1,9):
    num_glue_tasks[f"numglue_type_{num}"] = partial(
        TransformedDataset,
        data_path="json",
        data_name={
            "test": os.path.join(dir_path, f"NumGLUE-Type_{num}.jsonl"),
            },
        input_text=num_glue_input,
        output_text=num_glue_output,
        evaluation=num_glue_eval,
        test_split="test",
        fewshot_split="test",
    )

if __name__ == "__main__":

    for num in range(1,9):
        dataset = num_glue_tasks[f'type-{num}']()
        print(f"Type-{num} Dataset: {len(dataset)}")
        _input, _output = dataset.__getitem__(0)
        print("#### Input ###")
        print(_input)
        print("#### Output ###")
        print(_output)
