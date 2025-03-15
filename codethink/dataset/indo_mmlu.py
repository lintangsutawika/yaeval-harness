import re
from functools import partial

from codethink.dataset import register_task

from codethink._task import Task, create_task
from codethink._data import TransformedDataset

from codethink.utils import is_runnable_code
from codethink.dataset.utils import remove_boxed, last_boxed_only_string


def indo_mmlu_description(x):
    return f"Ini adalah soal {x['subject']} untuk {x['level']}. Pilihlah salah satu jawaban yang dianggap benar!"

def indo_mmlu_input(x):
    choices = "\n".join(eval(x["options"]))
    return f"{x['question']}\n{choices}"

def indo_mmlu_output(x):
    letter_answer = x["answer"]
    try:
        string_answer = eval(x["options"])[ord(letter_answer) - ord("A")].split(".")[-1].strip()
    except:
        string_answer = "None"
    return f"{letter_answer}:::{string_answer}"

def indo_mmlu_eval(prediction, ground_truth):
    if ":::" in ground_truth:
        letter_ans, string_ans = ground_truth.split(":::")
    else:
        string_ans = ground_truth
        letter_ans = "None"
    if prediction == letter_ans:
        return 1
    elif prediction == string_ans:
        return 1
    elif prediction.lower() == letter_ans.lower():
        return 1
    elif prediction.lower() == string_ans.lower():
        return 1
    
    full_ground_truth = f"{letter_ans}. {string_ans}"
    if prediction == full_ground_truth:
        return 1

    return 0

IndoMMLUDataset = partial(
    TransformedDataset,
    data_path="indolem/IndoMMLU",
    input_text=indo_mmlu_input,
    output_text=indo_mmlu_output,
    test_split="test",
)

def postprocess_reasoning(x, state):
    x = x["response"][0]
    _x = remove_boxed(last_boxed_only_string(x))
    if _x != "None":
        x = _x

    if "adalah" in x:
        x = x.split("adalah")[-1].strip()
    elif "answer is" in x:
        x = x.split("answer is")[-1].strip()
    
    # Try split \n\n
    x_split = x.split("\n\n")
    for _x in x_split:
        for alphabet in "ABCDE":
            if alphabet in _x:
                x = _x
                break
    if x.endswith("."):
        x = x[:-1]

    return x, state

@register_task(
    "indo_mmlu",
    dataset=IndoMMLUDataset,
    postprocessor=postprocess_reasoning,
    evaluation={"accuracy": indo_mmlu_eval},
    )
class IndoMMLUTask(Task):
    pass

@register_task(
    "indo_mmlu_original",
    dataset=partial(
        IndoMMLUDataset,
        input_text=lambda x: indo_mmlu_description(x) + "\n" + indo_mmlu_input(x),
        ),
    postprocessor=postprocess_reasoning,
    evaluation={"accuracy": indo_mmlu_eval},
    )
class IndoMMLUOriginalTask(Task):
    pass

@register_task(
    "indo_mmlu_cqr",
    dataset=partial(
        IndoMMLUDataset,
        input_text=lambda x: x["question"],
        ),
    postprocessor=postprocess_reasoning,
    evaluation={"accuracy": indo_mmlu_eval},
    )
class IndoMMLUCQRTask(Task):
    pass


if __name__ == "__main__":

    dataset = IndoMMLUDataset(
        num_fewshot=5,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)
