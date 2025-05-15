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

def input_fn(x):
    text_choice = x["choices"]
    choice_list = [f"{a}. {b}" for a,b in list(zip(letter_choice, text_choice))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either {",".join(letter_choice)}.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def zero_shot_input_fn(x):
    return input_fn(x)+"Let's think step by step."

def output_fn(x):
    text_choice = x["choices"]
    label = x['answer']
    letter = letter_choice[label]
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

@register_task("diff_strat_mmlu")
class DiffStratMMLUPro(YevalTask):
    data_path="cais/mmlu"
    data_name="all"
    input_text=zero_shot_input_fn
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

@register_task("diff_strat_mmlu_high_school_mathematics")
class DiffStratMMLUHighSchoolMathematics(DiffStratMMLUPro):
    data_name="high_school_mathematics"

@register_task("diff_strat_mmlu_college_mathematics")
class DiffStratMMLUCollegeMathematics(DiffStratMMLUPro):
    data_name="college_mathematics"

@register_task("diff_strat_mmlu_high_school_biology")
class DiffStratMMLUHighSchoolBiology(DiffStratMMLUPro):
    data_name="high_school_biology"

@register_task("diff_strat_mmlu_college_biology")
class DiffStratMMLUCollegeBiology(DiffStratMMLUPro):
    data_name="college_biology"

@register_task("diff_strat_mmlu_high_school_chemistry")
class DiffStratMMLUHighSchoolChemistry(DiffStratMMLUPro):
    data_name="high_school_chemistry"

@register_task("diff_strat_mmlu_college_chemistry")
class DiffStratMMLUCollegeChemistry(DiffStratMMLUPro):
    data_name="college_chemistry"

@register_task("diff_strat_mmlu_high_school_physics")
class DiffStratMMLUHighSchoolPhysics(DiffStratMMLUPro):
    data_name="high_school_physics"

@register_task("diff_strat_mmlu_college_physics")
class DiffStratMMLUCollegePhysics(DiffStratMMLUPro):
    data_name="college_physics"

@register_task("diff_strat_mmlu_high_school_computer_science")
class DiffStratMMLUHighSchoolComputerScience(DiffStratMMLUPro):
    data_name="high_school_computer_science"

@register_task("diff_strat_mmlu_college_computer_science")
class DiffStratMMLUComputerScience(DiffStratMMLUPro):
    data_name="college_computer_science"

@register_task("diff_strat_mmlu_nutrition")
class DiffStratMMLUNutrition(DiffStratMMLUPro):
    data_name="nutrition"



if __name__ == "__main__":
    pass
