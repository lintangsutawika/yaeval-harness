from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage
from yeval.response import (
        match_routing,
        preprocess_routing,
        postprocess_routing
        )

from functools import partial
from yeval.metrics.pass_at_k import classical_pass_at_k, openai_pass_at_k

letter_choice = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def input_fn(x):
    text_choice = x["options"]
    choice_list = [f"{a}. {b}" for a,b in list(zip(letter_choice, text_choice))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either {",".join(letter_choice)}.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def output_fn(x):
    text_choice = x["options"]
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

@register_task("diff_strat_mmlu_pro")
class DiffStratMMLUPro(YevalTask):
    data_path="TIGER-Lab/MMLU-Pro"
    input_text=input_fn
    output_text=output_fn
    postprocessor=postprocess
    eval_at_k=True
    sampling_args={"n":20}
    test_split="test"
    few_shot_split="validation"
    evaluation={
        # "pass@1": partial(classical_pass_at_k, k=1, metric_fn=eval_fn),
        # "pass@10": partial(classical_pass_at_k, k=10, metric_fn=eval_fn),
        "pass@1": partial(openai_pass_at_k, k=1, metric_fn=eval_fn),
        "pass@10": partial(openai_pass_at_k, k=10, metric_fn=eval_fn)
        }

# def crq_input_fn(x):
#     question = x['question']
#     return "Answer by thinking step-by-step and write the final answer in \\boxed\{\}. Question:\n"+question+"\nAnswer:"

# def crq_output_fn(x):
#     letter_choice = ["A", "B", "C", "D"]
#     text_choice = x["choices"]
#     label = x['answer']
#     text = text_choice[label]
#     letter = letter_choice[label]
#     return text

# from yeval.response.math_responses import (
#         last_boxed_only_string,
#         get_boxed_answer
#         )
# from yeval.metrics.math_eval import math_eval

# @register_task("crq_mmlu_redux_high_school_mathematics")
# class CRQMMLUReduxHighSchoolMathematics(DiffStratMMLURedux):
#     data_name="high_school_mathematics"
#     input_text=crq_input_fn
#     output_text=crq_output_fn
#     postprocessor=get_boxed_answer
#     evaluation={
#         "pass@1": partial(openai_pass_at_k, k=1, metric_fn=math_eval),
#         "pass@10": partial(openai_pass_at_k, k=10, metric_fn=math_eval)
#         }

# @register_task("crq_mmlu_redux_college_mathematics")
# class CRQMMLUReduxCollegeMathematics(CRQMMLUReduxHighSchoolMathematics):
#     data_name="college_mathematics"

# @register_task("crq_mmlu_redux_high_school_chemistry")
# class CRQMMLUReduxHighSchoolChemistry(CRQMMLUReduxHighSchoolMathematics):
#     data_name="high_school_chemistry"

# @register_task("crq_mmlu_redux_college_chemistry")
# class CRQMMLUReduxCollegeChemistry(CRQMMLUReduxHighSchoolMathematics):
#     data_name="college_chemistry"

# @register_task("crq_mmlu_redux_high_school_physics")
# class CRQMMLUReduxHighSchoolPhysics(CRQMMLUReduxHighSchoolMathematics):
#     data_name="high_school_physics"

# @register_task("crq_mmlu_redux_college_physics")
# class CRQMMLUReduxCollegePhysics(CRQMMLUReduxHighSchoolMathematics):
#     data_name="college_physics"

# @register_task("shuffle_mmlu_redux_college_physics")
# class ShuffleReduxCollegePhysicsA(DiffStratMMLUReduxCollegePhysics):
#     preprocessing=shuffle_choice
#     data_kwargs={"keep_in_memory": True}
#     sampling_args={"n":1}
#     eval_at_k=False
#     evaluation={
#         # "pass@1": partial(classical_pass_at_k, k=1, metric_fn=eval_fn),
#         # "pass@10": partial(classical_pass_at_k, k=10, metric_fn=eval_fn),
#         "accuracy": eval_fn,
#     }

if __name__ == "__main__":
    pass
