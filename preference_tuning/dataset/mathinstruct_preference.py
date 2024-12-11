# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

from functools import partial
from typing import Any, Dict, Mapping, Optional

from torchtune.data import Message
from torchtune.datasets._preference import PreferenceDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

dir_path = os.path.dirname(os.path.realpath(__file__))

def preprocessing(dataset, solution_type="PoT"):
    return dataset.filter(lambda example: solution_type in example["source"])

def pot_input(x):
    instruction = x['instruction']
    for keyword in ["Let", "Please"]:
        if keyword in instruction:
            index = max([m.start() for m in re.finditer(keyword, instruction)])
            break
    return instruction[:index].strip()

def pot_output(x):
    solution = x["output"]

    if "return" in solution:
        return solution

    lines = solution.split("\n")
    if "print" in lines[-1]:
        lines[-1] = lines[-1].replace("print(", "return ").replace(")", "")
    update_solution = ["def solution():\n"]+["    "+line for line in lines]
    return "\n".join(update_solution)

def pot_eval(prediction, ground_truth):
    answer = eval(ground_truth)
    if prediction == answer:
        return 1
    else:
        return 0

def cot_output(x, choice_keyword="Answer Choices:", output_keyword="answer is"):
    def _get_letters(input_string):
        pattern = r'\([a-zA-Z]+\)'
        matches = re.findall(pattern, input_string)
        return matches

    def _get_choices(markers, input_string):
        sections = {}
        for i, marker in enumerate(markers):
            start_index = input_string.find(marker)
            if start_index == -1:
                continue
            end_index = input_string.find(markers[i + 1]) if i + 1 < len(markers) else len(input_string)
            letter = marker.split("(")[-1].split(")")[0]
            sections[letter] = input_string[start_index + len(marker):end_index].strip()
        return sections

    instruction = x["instruction"]
    answer_dict = {}
    if choice_keyword in instruction:
        choice_string = instruction.split(choice_keyword)[-1].strip()
        letter_choice = _get_letters(choice_string)
        answer_dict = _get_choices(letter_choice, choice_string)

    output = x["output"]
    output = output.replace("answe ", "answer ")
    alt_answer = None
    if output_keyword in output:
        answer = output.split(output_keyword)[-1].strip()
        if answer == '':
            return output, None
        else:
            if answer[-1] == ".":
                answer = answer[:-1]
            if "Option" in answer:
                answer = answer.split("Option")[-1].strip()
            if "\n" in answer:
                answer = answer.split("\n")[0].strip()
            if answer_dict != {}:
                for a in answer_dict.values():
                    if a in answer:
                        answer = answer.replace(a, "").strip()
                if answer in answer_dict.keys():
                    alt_answer = answer_dict[answer]
            return answer, alt_answer
    else:
        return output, None

def cot_eval(prediction, ground_truth):

    def _extract_number(input_string):
        match = re.search(r'\d+\.?\d*', input_string)
        if match:
            number = match.group()
            return number
            # return float(number) if '.' in number else int(number)
        return None  # Return None if no number is found

    score = 0
    for answer in eval(ground_truth):
        if answer is None:
            continue

        if prediction == answer:
            score = 1
        else:
            try:
                answer = eval(answer)
                if eval(prediction) == answer:
                    score = 1
            except Exception as E:
                pass

            try:
                answer = eval(_extract_number(answer))
                if eval(prediction) == answer:
                    score = 1
            except Exception as E:
                pass
    return score

MathInstructPoTDataset = partial(
    TransformedDataset,
    data_path="TIGER-Lab/MathInstruct",
    preprocessing=partial(preprocessing, solution_type="PoT"),
    input_text=pot_input,
    output_text=pot_output,
    evaluation=lambda x, y: -1,
    test_split="train",
)

MathInstructCoTDataset = partial(
    TransformedDataset,
    data_path="TIGER-Lab/MathInstruct",
    preprocessing=partial(preprocessing, solution_type="CoT"),
    input_text=lambda x: x["instruction"],
    output_text=cot_output,
    evaluation=cot_eval,
    test_split="train",
)

class PairedMessages(Transform):

    def __init__(
        self, train_on_input: bool = False, column_map: Optional[Dict[str, str]] = None
    ):
        self.train_on_input = train_on_input
        self._column_map = column_map

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        column_map = self._column_map or {}
        key_prompt = column_map.get("prompt", "prompt")
        key_chosen = column_map.get("chosen", "chosen")
        key_rejected = column_map.get("rejected", "rejected")

        chosen_messages = [
            Message(
                role="user", content=sample[key_prompt], masked=not self.train_on_input
            ),
            Message(role="assistant", content=sample[key_chosen]),
        ]

        rejected_messages = [
            Message(
                role="user", content=sample[key_prompt], masked=not self.train_on_input
            ),
            Message(role="assistant", content=sample[key_rejected]),
        ]

        return {"chosen": chosen_messages, "rejected": rejected_messages}


def mathinstruct_preference(
    tokenizer: ModelTokenizer,
    *,
    source: str = None,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> PreferenceDataset:
    """
    Family of preference datasets similar to the `Stack Exchange Paired dataset
    <https://huggingface.co/datasets/lvwerra/stack-exchange-paired>`_.

    It is recommended to configure the tokenizer with the :class:`~torchtune.data.QuestionAnswerTemplate`
    in conjunction with this dataset.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is ``lvwerra/stack-exchange-paired``.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "prompt",
            "chosen", and "rejected" column names to the actual column names in the dataset.
            Keys should be "prompt", "chosen", and "rejected" and values should be the actual column names.
            Default is None, keeping the default column names.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Returns:
        PreferenceDataset: The preference dataset built from source paired data.
    """

    column_map = column_map or {
        "prompt": "question",
        "chosen": "response_i",
        "rejected": "response_j",
    }

    message_transform = PairedMessages(
        train_on_input=train_on_input, column_map=column_map
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    source = "csv"
    load_dataset_kwargs = {
        # "data_files": {"train": os.path.join(dir_path, "mathinstruct_preference.csv")}
        "data_files": {"train": os.path.join(dir_path, "mathinstruct_cot.csv")}
        }

    return PreferenceDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        split=split,
        data_dir="data/rl",
        **load_dataset_kwargs,
    )


if __name__ == "__main__":
    pass
