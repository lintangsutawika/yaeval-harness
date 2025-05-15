import os
from yeval.task import register_task, YevalTask

dir_path = os.path.dirname(os.path.realpath(__file__))

def tabmwp_input(x):
    return "Table:\n"+x['table']+"\nQuestion:\n"+x['question'].strip()+"\nAnswer:"

def tabmwp_output(x):
    return x["answer"]

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

@register_task("tabmwp")
class TabMWPTask(YevalTask):
    data_path="json"
    data_kwargs={
        "data_files": {
            "test": os.path.join(dir_path, "tabmwp_test.jsonl"),
            "dev": os.path.join(dir_path, "tabmwp_dev.jsonl"),
            }
        }
    input_text=tabmwp_input
    output_text=tabmwp_output
    test_split="test"
    evaluation={"accuracy": tabmwp_eval}
