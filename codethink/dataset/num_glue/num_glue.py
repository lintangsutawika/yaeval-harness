import os
from functools import partial

from codethink.dataset import register_task

from codethink._task import Task, create_task
from codethink._data import TransformedDataset

from codethink.utils import is_runnable_code

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
    num_glue_tasks[f'type-{num}'] = partial(
        TransformedDataset,
        data_path="json",
        data_name={
            "test": os.path.join(dir_path, f"NumGLUE-Type_{num}.jsonl"),
        },
        input_text=num_glue_input,
        output_text=num_glue_output,
        test_split="test",
        fewshot_split="test",
    )

def preprocess_routing(x, state):
    current_step = state["current_step"]
    solve_with = state["step"][current_step-1]["output"]
    if "programming language" in solve_with:
        state["system_message"] = "code"
    # elif "natural language" in solve_with:
    else:
        state["system_message"] = "cot"
    return x, state

def postprocess_routing(x, state):
    x = x["response"][0]
    current_step = state["current_step"]
    solve_with = state["step"][current_step-1]["output"]
    if "programming language" in solve_with:
        x = is_runnable_code(x) 
    # elif "natural language" in solve_with:
    else:
        try:
            x = x.split("answer is")[-1].strip()
        except:
            pass
    return x, state

def postprocess_PL_or_NL(x, state):
    x = x["response"][0]
    exec_result = is_runnable_code(x) 
    if exec_result:
        return exec_result, state
    else:
        try:
            x = x.split("answer is")[-1].strip()
        except:
            pass
    return x, state

@register_task(
    "numglue_type_1",
    dataset=num_glue_tasks["type-1"],
    evaluation={"accuracy": num_glue_eval},
    postprocessor=postprocess_PL_or_NL,
)
class NumGLUEType1(Task):
    pass

@register_task(
    "numglue_type_2",
    dataset=num_glue_tasks["type-2"],
    evaluation={"accuracy": num_glue_eval},
    postprocessor=postprocess_PL_or_NL,
)
class NumGLUEType2(Task):
    pass

@register_task(
    "numglue_type_3",
    dataset=num_glue_tasks["type-3"],
    evaluation={"accuracy": num_glue_eval},
    postprocessor=postprocess_PL_or_NL,
)
class NumGLUEType3(Task):
    pass

@register_task(
    "numglue_type_4",
    dataset=num_glue_tasks["type-4"],
    evaluation={"accuracy": num_glue_eval},
    postprocessor=postprocess_PL_or_NL,
)
class NumGLUEType4(Task):
    pass

@register_task(
    "numglue_type_5",
    dataset=num_glue_tasks["type-5"],
    evaluation={"accuracy": num_glue_eval},
    postprocessor=postprocess_PL_or_NL,
)
class NumGLUEType5(Task):
    pass

@register_task(
    "numglue_type_6",
    dataset=num_glue_tasks["type-6"],
    evaluation={"accuracy": num_glue_eval},
    postprocessor=postprocess_PL_or_NL,
)
class NumGLUEType6(Task):
    pass

@register_task(
    "numglue_type_7",
    dataset=num_glue_tasks["type-7"],
    evaluation={"accuracy": num_glue_eval},
    postprocessor=postprocess_PL_or_NL,
)
class NumGLUEType7(Task):
    pass

@register_task(
    "numglue_type_8",
    dataset=num_glue_tasks["type-8"],
    evaluation={"accuracy": num_glue_eval},
    postprocessor=postprocess_PL_or_NL,
)
class NumGLUEType8(Task):
    pass

@register_task(
    "numglue_type_1_routing_pl_first",
    subtask_list=[
        Task(
            name="numglue_type_1_routing",
            dataset=partial(
                num_glue_tasks["type-1"],
                input_text=lambda x: num_glue_input(x).replace("\nAnswer:", "")+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
        ),
        Task(
            name="numglue_type_1",
            dataset=num_glue_tasks["type-1"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": num_glue_eval},
        ),
    ])
class NumGLUEType1Routing(Task):
    pass

@register_task(
    "numglue_type_2_routing_pl_first",
    subtask_list=[
        Task(
            name="numglue_type_2_routing",
            dataset=partial(
                num_glue_tasks["type-2"],
                input_text=lambda x: num_glue_input(x).replace("\nAnswer:", "")+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
        ),
        Task(
            name="numglue_type_2",
            dataset=num_glue_tasks["type-2"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": num_glue_eval},
        ),
    ])
class NumGLUEType2Routing(Task):
    pass

@register_task(
    "numglue_type_3_routing_pl_first",
    subtask_list=[
        Task(
            name="numglue_type_3_routing",
            dataset=partial(
                num_glue_tasks["type-3"],
                input_text=lambda x: num_glue_input(x).replace("\nAnswer:", "")+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
        ),
        Task(
            name="numglue_type_3",
            dataset=num_glue_tasks["type-3"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": num_glue_eval},
        ),
    ])
class NumGLUEType3Routing(Task):
    pass

@register_task(
    "numglue_type_4_routing_pl_first",
    subtask_list=[
        Task(
            name="numglue_type_4_routing",
            dataset=partial(
                num_glue_tasks["type-4"],
                input_text=lambda x: num_glue_input(x).replace("\nAnswer:", "")+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
        ),
        Task(
            name="numglue_type_4",
            dataset=num_glue_tasks["type-4"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": num_glue_eval},
        ),
    ])
class NumGLUEType4Routing(Task):
    pass

@register_task(
    "numglue_type_5_routing_pl_first",
    subtask_list=[
        Task(
            name="numglue_type_5_routing",
            dataset=partial(
                num_glue_tasks["type-5"],
                input_text=lambda x: num_glue_input(x).replace("\nAnswer:", "")+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
        ),
        Task(
            name="numglue_type_5",
            dataset=num_glue_tasks["type-5"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": num_glue_eval},
        ),
    ])
class NumGLUEType5Routing(Task):
    pass

@register_task(
    "numglue_type_6_routing_pl_first",
    subtask_list=[
        Task(
            name="numglue_type_6_routing",
            dataset=partial(
                num_glue_tasks["type-6"],
                input_text=lambda x: num_glue_input(x).replace("\nAnswer:", "")+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
        ),
        Task(
            name="numglue_type_6",
            dataset=num_glue_tasks["type-6"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": num_glue_eval},
        ),
    ])
class NumGLUEType6Routing(Task):
    pass

@register_task(
    "numglue_type_7_routing_pl_first",
    subtask_list=[
        Task(
            name="numglue_type_7_routing",
            dataset=partial(
                num_glue_tasks["type-7"],
                input_text=lambda x: num_glue_input(x).replace("\nAnswer:", "")+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
        ),
        Task(
            name="numglue_type_7",
            dataset=num_glue_tasks["type-7"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": num_glue_eval},
        ),
    ])
class NumGLUEType7Routing(Task):
    pass

@register_task(
    "numglue_type_8_routing_pl_first",
    subtask_list=[
        Task(
            name="numglue_type_8_routing",
            dataset=partial(
                num_glue_tasks["type-8"],
                input_text=lambda x: num_glue_input(x).replace("\nAnswer:", "")+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
        ),
        Task(
            name="numglue_type_8",
            dataset=num_glue_tasks["type-8"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": num_glue_eval},
        ),
    ])
class NumGLUEType8Routing(Task):
    pass

if __name__ == "__main__":

    for num in range(1,9):
        dataset = num_glue_tasks[f'type-{num}']()
        print(f"Type-{num} Dataset: {len(dataset)}")
        _input, _output = dataset.__getitem__(0)
        print("#### Input ###")
        print(_input)
        print("#### Output ###")
        print(_output)
