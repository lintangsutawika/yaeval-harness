from functools import partial
from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

letter_choice = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def input_fn(x):
    text_choice = x["options"]
    choice_list = [f"{a}. {b}" for a,b in list(zip(letter_choice, text_choice))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either {", ".join(letter_choice[:-1])}, or {letter_choice[-1]}.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def output_fn(x):
    text_choice = x["options"]
    letter = x['answer']
    label = letter_choice.index(letter)
    text = text_choice[label]
    return [letter, text, f"{letter}. {text}"]

def check_answer(prediction, ground_truth):
    score = 0
    try:
        letter, text, full_span = ground_truth
        if full_span == ground_truth:
            return 1
        prediction = prediction.split(".")[0]
        if prediction in letter_choice:
            if prediction == letter:
                score = 1
        elif prediction == text:
            score = 1
    except Exception as e:
        pass
    return score

def eval_fn(prediction, ground_truth):
    score = check_answer(prediction, ground_truth)
    if score == 0:
        for gt in ground_truth:
            score = math_eval(prediction, gt)
            if score == 1:
                break
    return score

@register_task("mmlu_pro")
class MMLUProTask(YevalTask):
    data_path="TIGER-Lab/MMLU-Pro"
    input_text=input_fn
    output_text=output_fn
    test_split="test"
    evaluation={"accuracy": eval_fn}

if __name__ == "__main__":
    pass
