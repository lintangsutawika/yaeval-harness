import os
import re
from functools import partial

from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage
from yeval.response import (
    match_routing,
    preprocess_routing,
    postprocess_routing
    )

from yeval.metrics import math_eval

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
        if prediction.endswith("."):
            prediction = prediction[:-1]
        if prediction.startswith(":"):
            prediction = prediction[1:]
        prediction = prediction.strip()
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

class NumGLUEBaseTask(YevalTask):
    data_path="json"
    input_text=num_glue_input
    output_text=num_glue_output
    test_split="train"
    evaluation={"accuracy": math_eval}
    logging=log_token_usage

class NumGLUERoutingPLFirstTask(NumGLUEBaseTask):
    input_text=lambda x: num_glue_input(x).replace("\nAnswer:", "")+"\n\nWhich method is the best way to solve this problem?"
    output_text=lambda x: -1
    system_message="select_pl_first"
    sampling_args={"stop": ["\n\n", "\n"]}
    evaluation={"accuracy": lambda x,y: -1}

class NumGLUERoutingNLFirstTask(NumGLUERoutingPLFirstTask):
    system_message="select_nl_first"

class NumGLUERoutingStage(NumGLUEBaseTask):
    preprocessor=preprocess_routing
    postprocessor=postprocess_routing

### NumGLUE TYPE 1
class NumGLUEType1Task(NumGLUEBaseTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_1.jsonl")

class NumGLUEType1RoutingPLFirstTask(NumGLUERoutingPLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_1.jsonl")

class NumGLUEType1RoutingNLFirstTask(NumGLUERoutingNLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_1.jsonl")

class NumGLUEType1RoutingStage(NumGLUERoutingStage):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_1.jsonl")

@register_task("numglue_type_1_routing_pipeline_pl_first")
class NumGLUEType1RoutingATask(YevalTask):
    subtask_list=[
        NumGLUEType1RoutingPLFirstTask,
        NumGLUEType1RoutingStage
    ]

@register_task("numglue_type_1_routing_pipeline_nl_first")
class NumGLUEType1RoutingBTask(YevalTask):
    subtask_list=[
        NumGLUEType1RoutingNLFirstTask,
        NumGLUEType1RoutingStage
    ]

### NumGLUE TYPE 2
class NumGLUEType2Task(NumGLUEBaseTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_2.jsonl")

class NumGLUEType2RoutingPLFirstTask(NumGLUERoutingPLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_2.jsonl")

class NumGLUEType2RoutingNLFirstTask(NumGLUERoutingNLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_2.jsonl")

class NumGLUEType2RoutingStage(NumGLUERoutingStage):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_2.jsonl")

@register_task("numglue_type_2_routing_pipeline_pl_first")
class NumGLUEType2RoutingATask(YevalTask):
    subtask_list=[
        NumGLUEType2RoutingPLFirstTask,
        NumGLUEType2RoutingStage
    ]

@register_task("numglue_type_2_routing_pipeline_nl_first")
class NumGLUEType2RoutingBTask(YevalTask):
    subtask_list=[
        NumGLUEType2RoutingNLFirstTask,
        NumGLUEType2RoutingStage
    ]

### NumGLUE TYPE 3
class NumGLUEType3Task(NumGLUEBaseTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_3.jsonl")

class NumGLUEType3RoutingPLFirstTask(NumGLUERoutingPLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_3.jsonl")

class NumGLUEType3RoutingNLFirstTask(NumGLUERoutingNLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_3.jsonl")

class NumGLUEType3RoutingStage(NumGLUERoutingStage):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_3.jsonl")

@register_task("numglue_type_3_routing_pipeline_pl_first")
class NumGLUEType3RoutingATask(YevalTask):
    subtask_list=[
        NumGLUEType3RoutingPLFirstTask,
        NumGLUEType3RoutingStage
    ]

@register_task("numglue_type_3_routing_pipeline_nl_first")
class NumGLUEType3RoutingBTask(YevalTask):
    subtask_list=[
        NumGLUEType3RoutingNLFirstTask,
        NumGLUEType3RoutingStage
    ]

### NumGLUE TYPE 4
class NumGLUEType4Task(NumGLUEBaseTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_4.jsonl")

class NumGLUEType4RoutingPLFirstTask(NumGLUERoutingPLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_4.jsonl")

class NumGLUEType4RoutingNLFirstTask(NumGLUERoutingNLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_4.jsonl")

class NumGLUEType4RoutingStage(NumGLUERoutingStage):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_4.jsonl")

@register_task("numglue_type_4_routing_pipeline_pl_first")
class NumGLUEType4RoutingATask(YevalTask):
    subtask_list=[
        NumGLUEType4RoutingPLFirstTask,
        NumGLUEType4RoutingStage
    ]

@register_task("numglue_type_4_routing_pipeline_nl_first")
class NumGLUEType4RoutingBTask(YevalTask):
    subtask_list=[
        NumGLUEType4RoutingNLFirstTask,
        NumGLUEType4RoutingStage
    ]

### NumGLUE TYPE 5
class NumGLUEType5Task(NumGLUEBaseTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_5.jsonl")

class NumGLUEType5RoutingPLFirstTask(NumGLUERoutingPLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_5.jsonl")

class NumGLUEType5RoutingNLFirstTask(NumGLUERoutingNLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_5.jsonl")

class NumGLUEType5RoutingStage(NumGLUERoutingStage):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_5.jsonl")

@register_task("numglue_type_5_routing_pipeline_pl_first")
class NumGLUEType5RoutingATask(YevalTask):
    subtask_list=[
        NumGLUEType5RoutingPLFirstTask,
        NumGLUEType5RoutingStage
    ]

@register_task("numglue_type_5_routing_pipeline_nl_first")
class NumGLUEType5RoutingBTask(YevalTask):
    subtask_list=[
        NumGLUEType5RoutingNLFirstTask,
        NumGLUEType5RoutingStage
    ]

### NumGLUE TYPE 6
class NumGLUEType6Task(NumGLUEBaseTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_6.jsonl")

class NumGLUEType6RoutingPLFirstTask(NumGLUERoutingPLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_6.jsonl")

class NumGLUEType6RoutingNLFirstTask(NumGLUERoutingNLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_6.jsonl")

class NumGLUEType6RoutingStage(NumGLUERoutingStage):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_6.jsonl")

@register_task("numglue_type_6_routing_pipeline_pl_first")
class NumGLUEType6RoutingATask(YevalTask):
    subtask_list=[
        NumGLUEType6RoutingPLFirstTask,
        NumGLUEType6RoutingStage
    ]

@register_task("numglue_type_6_routing_pipeline_nl_first")
class NumGLUEType6RoutingBTask(YevalTask):
    subtask_list=[
        NumGLUEType6RoutingNLFirstTask,
        NumGLUEType6RoutingStage
    ]

### NumGLUE TYPE 7
class NumGLUEType7Task(NumGLUEBaseTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_7.jsonl")

class NumGLUEType7RoutingPLFirstTask(NumGLUERoutingPLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_7.jsonl")

class NumGLUEType7RoutingNLFirstTask(NumGLUERoutingNLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_7.jsonl")

class NumGLUEType7RoutingStage(NumGLUERoutingStage):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_7.jsonl")

@register_task("numglue_type_7_routing_pipeline_pl_first")
class NumGLUEType7RoutingATask(YevalTask):
    subtask_list=[
        NumGLUEType7RoutingPLFirstTask,
        NumGLUEType7RoutingStage
    ]

@register_task("numglue_type_7_routing_pipeline_nl_first")
class NumGLUEType7RoutingBTask(YevalTask):
    subtask_list=[
        NumGLUEType7RoutingNLFirstTask,
        NumGLUEType7RoutingStage
    ]

### NumGLUE TYPE 8
class NumGLUEType8Task(NumGLUEBaseTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_8.jsonl")

class NumGLUEType8RoutingPLFirstTask(NumGLUERoutingPLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_8.jsonl")

class NumGLUEType8RoutingNLFirstTask(NumGLUERoutingNLFirstTask):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_8.jsonl")

class NumGLUEType8RoutingStage(NumGLUERoutingStage):
    data_name=os.path.join(dir_path, f"NumGLUE-Type_8.jsonl")

@register_task("numglue_type_8_routing_pipeline_pl_first")
class NumGLUEType8RoutingATask(YevalTask):
    subtask_list=[
        NumGLUEType8RoutingPLFirstTask,
        NumGLUEType8RoutingStage
    ]

@register_task("numglue_type_8_routing_pipeline_nl_first")
class NumGLUEType8RoutingBTask(YevalTask):
    subtask_list=[
        NumGLUEType8RoutingNLFirstTask,
        NumGLUEType8RoutingStage
    ]
