from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage
from yeval.response import (
        match_routing,
        preprocess_routing,
        postprocess_routing
        )

from functools import partial
from yeval.metrics.pass_at_k import classical_pass_at_k, openai_pass_at_k

def input_fn(x):
    letter_choice = ["A", "B", "C", "D"]
    text_choice = x["choices"]
    choice_list = [f"{a}. {b}" for a,b in list(zip(letter_choice, text_choice))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either A, B, C, D.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def output_fn(x):
    letter_choice = ["A", "B", "C", "D"]
    text_choice = x["choices"]
    label = x['answer']
    text = text_choice[label]
    letter = letter_choice[label]
    return [letter, text, f"{letter}. {text}"]

def eval_fn(prediction, ground_truth):
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

class MMLURedux(YevalTask):
    data_path="edinburgh-dawg/mmlu-redux"
    data_name="high_school_mathematics"
    input_text=input_fn
    output_text=output_fn
    eval_at_k=True
    sampling_args={"n":20}
    test_split="test"
    evaluation={
        # "pass@1": partial(classical_pass_at_k, k=1, metric_fn=eval_fn),
        # "pass@10": partial(classical_pass_at_k, k=10, metric_fn=eval_fn),
        "pass@1": partial(openai_pass_at_k, k=1, metric_fn=eval_fn),
        "pass@10": partial(openai_pass_at_k, k=10, metric_fn=eval_fn)
        }


if __name__ == "__main__":
    pass
