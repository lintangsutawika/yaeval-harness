import random

from typing import Union, Callable
from functools import partial
from torch.utils.data import Dataset

from datasets import load_dataset

class TransformedDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 data_name: str=None,
                 name: str=None,
                 input_text: Union[str, Callable]=None,
                 output_text: Union[str, Callable]=None,
                 test_split: str=None,
                 fewshot_input_text: Union[str, Callable]=None,
                 fewshot_output_text: Union[str, Callable]=None,
                 fewshot_split: str=None,
                 num_fewshot: int=0,
                 sampler: str=None,
                 fewshot_delimiter: str="\n\n",
                 answer_delimiter: str="\n",
                 ):
        
        if name is None:
            if data_name is None:
                self.name = f"{data_path}"
            else:
                self.name = f"{data_path}-{data_name}"
        else:
            self.name = name

        self.dataset = load_dataset(
            path=data_path,
            name=data_name,
        )
        self.test_split = test_split
        self.fewshot_split = fewshot_split
        self.num_fewshot = num_fewshot
        self.sampler = sampler
        self.fewshot_delimiter = fewshot_delimiter
        self.answer_delimiter = answer_delimiter

        self.use_fewshot_input = False
        if fewshot_input_text is not None:
            self.use_fewshot_input = True
        
        self.use_fewshot_output = False
        if fewshot_output_text is not None:
             self.use_fewshot_output = True

        def _transform(example, fn, feature, **kwargs):
            if isinstance(fn, str):
                try:
                    example[feature] = example[fn]
                except Exception as e:
                    raise e
            elif callable(fn):
                example[feature] = fn(example, **kwargs)
            return example

        all_split = [self.test_split]
        if self.fewshot_split is not None:
            all_split.append(self.fewshot_split)

        for split in all_split:

            
            if self.use_fewshot_input:
                self.dataset[split] = self.dataset[split].map(partial(_transform, fn=fewshot_input_text, feature="__fewshot_input__"))

            if self.use_fewshot_output:
                self.dataset[split] = self.dataset[split].map(partial(_transform, fn=fewshot_output_text, feature="__fewshot_output__"))

            self.dataset[split] = self.dataset[split].map(partial(_transform, fn=input_text, feature="__input__"))
            self.dataset[split] = self.dataset[split].map(partial(_transform, fn=output_text, feature="__output__"))

        if self.num_fewshot > 0:
            if self.sampler is None:
                sample_idx = random.sample(
                    list(range(len(self.dataset[self.fewshot_split]))),
                    self.num_fewshot
                )
            elif self.sampler == "first_n":
                sample_idx = list(range(self.num_fewshot))
            else:
                raise NotImplemented
            
            fewshot_samples = self.get_fewshot(sample_idx)
            self.dataset[test_split] = self.dataset[test_split].map(partial(_transform, fn=self.prepend_fewshot, feature="__input__", fewshot_samples=fewshot_samples))

    def get_fewshot(self, sample_idx):
        fewshot_samples = []

        input_feature = "__fewshot_input__" if self.use_fewshot_input else "__input__"
        output_feature = "__fewshot_output__" if self.use_fewshot_output else "__output__"
        for idx in sample_idx:
            sample = self.dataset[self.fewshot_split][idx]
            fewshot_samples.append(
                f"{self.answer_delimiter}".join([
                    str(sample[input_feature]),
                    str(sample[output_feature])
                ])
            )
        return fewshot_samples

    def prepend_fewshot(self, example, fewshot_samples):
        fewshot_samples.append(str(example["__input__"]))
        return f"{self.fewshot_delimiter}".join(fewshot_samples)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return (
            self.dataset[self.test_split][i]["__input__"],
            self.dataset[self.test_split][i]["__output__"]
        )


if __name__ == "__main__":

    import re

    def gsm8k_output(x):
        answer = x["answer"]
        answer = answer.split("#### ")[-1]
        answer = float(re.findall(r'\d+', answer)[0])
        return answer

    def gsm8k_fewshot_output(x):
        answer = x["answer"]
        answer = re.sub("####", "So the answer is", answer)
        return answer

    gsm8k_dataset = TransformedDataset(
        data_path="gsm8k",
        data_name="main",
        input_text="question",
        output_text=gsm8k_output,
        fewshot_output_text=gsm8k_fewshot_output,
        test_split="test",
        fewshot_split="train",
        num_fewshot=5,
        sampler=None,
    )

    input, output = gsm8k_dataset.__getitem__(0)
    print("#### Input ###")
    print(input)
    print("#### Output ###")
