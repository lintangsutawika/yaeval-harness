from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage
from yeval.response import (
        match_routing,
        preprocess_routing,
        postprocess_routing
        )

from functools import partial
from yeval.metrics.pass_at_k import classical_pass_at_k, openai_pass_at_k

letter_choice = ["A", "B", "C", "D"]
option_keys = [f"option_{i}" for i in ["a", "b", "c", "d"]]

def input_fn(x):
    text_choice = [x[key] for key in option_keys]
    choice_list = [f"{a}. {b}" for a,b in list(zip(letter_choice, text_choice))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either {",".join(letter_choice)}.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def zero_shot_input_fn(x):
    return input_fn(x)+"Let's think step by step."

def output_fn(x):
    text_choice = [x[key] for key in option_keys]
    letter = x['answer']
    label = letter_choice.index(letter)
    text = text_choice[label]
    return [letter, text, f"{letter}. {text}"]

def eval_fn(prediction, ground_truth):
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

def postprocess(x):
    answer_snippet = [
        "Answer:",
        "answer is:",
        "answer is",
    ]
    for snippet in answer_snippet:
        if snippet in x:
            x = x.split(snippet)[-1]
            x = x.strip()
            return x
    return x

class GlobalMMLUBaseTask(YevalTask):
    data_path="CohereForAI/Global-MMLU"
    input_text=zero_shot_input_fn
    output_text=output_fn
    postprocessor=postprocess
    test_split="test"
    evaluation={
        "accuracy": eval_fn
        # "pass@1": partial(classical_pass_at_k, k=1, metric_fn=eval_fn),
        # "pass@10": partial(classical_pass_at_k, k=10, metric_fn=eval_fn),
        # "pass@1": partial(openai_pass_at_k, k=1, metric_fn=eval_fn),
        # "pass@10": partial(openai_pass_at_k, k=10, metric_fn=eval_fn)
        }

@register_task("global_mmlu_id")
class GlobalMMLUIdTask(GlobalMMLUBaseTask):
    data_name="id"

if __name__ == "__main__":
    pass
