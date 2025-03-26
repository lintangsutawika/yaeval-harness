from yeval.task import register_task, YevalTask

from yeval.logging.usage import log_token_usage
from yeval.response import (
        match_routing,
        preprocess_routing,
        postprocess_routing
        )

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
    logging=log_token_usage

@register_task("commonsense_qa_routing_nl_first")
class CommonsenseQARoutingNLFirstTask(CommonsenseQATask):
    input_text=lambda x: x['question']+"\n\nWhich method is the best way to solve this problem?"
    output_text=lambda x: "natural language"
    system_message="select_nl_first"
    sampling_args={"stop": ["\n\n", "\n"]}
    evaluation={"accuracy": match_routing}

@register_task("commonsense_qa_routing_pl_first")
class CommonsenseQARoutingPLFirstTask(CommonsenseQATask):
   input_text=lambda x: x['question']+"\n\nWhich method is the best way to solve this problem?"
   output_text=lambda x: "natural language"
   system_message="select_pl_first"
   sampling_args={"stop": ["\n\n", "\n"]}
   evaluation={"accuracy": match_routing}

class CommonsenseQARoutingStage(CommonsenseQATask):
   name="solve_commonsense_qa"
   preprocessor=preprocess_routing
   postprocessor=postprocess_routing

@register_task("commonsense_qa_routing_nl_first")
class CommonsenseQARoutingATask(YevalTask):
   subtask_list=[
       CommonsenseQARoutingNLFirstTask,
       CommonsenseQARoutingStage
   ]

@register_task("commonsense_qa_routing_pl_first")
class CommonsenseQARoutingBTask(YevalTask):
   subtask_list=[
       CommonsenseQARoutingPLFirstTask,
       CommonsenseQARoutingStage
   ]


if __name__ == "__main__":
    pass
