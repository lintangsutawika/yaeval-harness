from yeval.task import register_task, YevalTask

def cqa_input(x):
    choices = x["choices"]
    choice_list = [f"{a}. {b}" for a,b in list(zip(choices["label"], choices["text"]))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either A, B, C, D or E.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def cqa_output(x):
    label = x['answerKey']
    label_idx = x['choices']['label'].index(label)
    text = x['choices']['text'][label_idx]
    return [label, text, f"{label}. {text}"]

def cqa_eval(prediction, ground_truth):
    score = 0
    try:
        letter, text, full_span = ground_truth
        if full_span == ground_truth:
            return 1
        prediction = prediction.split(".")[0]
        if prediction in ["A", "B", "C", "D", "E"]:
            if prediction == letter:
                score = 1
        elif prediction == text:
            score = 1
    except Exception as e:
        pass
    return score

@register_task("commonsense_qa")
class CommonsenseQATask(YevalTask):
    data_path="tau/commonsense_qa"
    input_text=cqa_input
    output_text=cqa_output
    test_split="validation"
    evaluation={"accuracy": cqa_eval}

if __name__ == "__main__":
    pass
