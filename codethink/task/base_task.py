import os
import numpy as np

from dataclasses import dataclass
from functools import partial
from typing import Union, Callable, Dict, List
from codethink._system_message import Prompt, SYSTEM_MESSAGE
from codethink.response import POSTPROCESS

from transformers import AutoTokenizer

def match_fn(x, y):
    try:
        return 1 if x == y else 0
    except Exception as e:
        return 0

def get_postprocess_fn(postprocess):
    if postprocess in POSTPROCESS:
        return POSTPROCESS[postprocess]
    else:
        return postprocess

@dataclass
class TaskConfig:
    name: str
    subtask_list: List['TaskConfig'] = None
    preprocessor: Callable = None
    postprocessor: Callable = None
    inference_fn: Callable = None
    prompt: str = None
    evaluation: Dict = None
    sampling_args: Dict = None
    callback: Callable = None
    name: str = None,
    subtask_list: list = None,
    # dataset = None,
    # dataset_kwargs: dict = {},
    preprocessor: callable = lambda x, y: (x, y),
    postprocessor: callable = lambda x, y: (x, y),
    inference_fn: callable = None,
    tokenizer: str = None,
    system_message: Union[str, Prompt] = None,
    evaluation: Union[str, Dict[str, Callable]]="match",
    sampling_args: dict = {},
    system_role: str = "system",
    logging: callable = None,
    sample_agg_fn: callable = np.mean,
    data_path: str=None,
    data_name: str=None,
    input_text: Union[str, Callable]=None,
    output_text: Union[str, Callable]=None,
    preprocessing: Callable=None,
    test_split: str=None,
    fewshot_input_text: Union[str, Callable]=None,
    fewshot_output_text: Union[str, Callable]=None,
    fewshot_split: str=None,
    num_fewshot: int=0,
    sampler: str=None,
    fewshot_delimiter: str="\n\n",
    answer_delimiter: str="\n",
    n_samples: Union[int, float]=None,
    data_kwargs: dict=None,
    batch_processing: bool=False,

# @dataclass
class YevalTask:

    name: str = None
    subtask_list: list = None
    # dataset = None,
    # dataset_kwargs: dict = {},
    preprocessor: callable = None
    postprocessor: callable = None
    inference_fn: callable = None
    tokenizer: str = None
    system_message: Union[str, Prompt] = None
    evaluation: Union[str, Dict[str, Callable]]="match"
    sampling_args: dict =None
    system_role: str = "system"
    logging: callable = None
    sample_agg_fn: callable = np.mean
    data_path: str=None
    data_name: str=None
    input_text: Union[str, Callable]=None
    output_text: Union[str, Callable]=None
    preprocessing: Callable=None
    test_split: str=None
    fewshot_input_text: Union[str, Callable]=None
    fewshot_output_text: Union[str, Callable]=None
    fewshot_split: str=None
    num_fewshot: int=0
    sampler: str=None
    fewshot_delimiter: str="\n\n"
    answer_delimiter: str="\n"
    n_samples: Union[int, float]=None
    data_kwargs: dict=None
    batch_processing: bool=False

    @staticmethod
    def _input_text(self, x):
        return self.input_text(x)


    def __init__(
        self,
        name: str = None,
        preprocessor: callable = None,
        postprocessor: callable = None,
        system_message: Union[str, Prompt] = None,
        system_role: str = "system",
        sampling_args: dict = None,
        num_fewshot: Union[int, float] = None,
        n_samples: Union[int, float] = None,
        ):

        if self.subtask_list is not None:
            self.subtask_list = [task() for task in self.subtask_list]

        if self.data_path is not None:
            from codethink.task import YevalDataset
            self.dataset = YevalDataset(
                data_path=self.data_path,
                data_name=self.data_name,
                input_text=self.input_text.__func__,
                output_text=self.output_text.__func__,
                preprocessing=self.preprocessing,
                test_split=self.test_split,
                fewshot_input_text=self.fewshot_input_text.__func__ if self.fewshot_input_text else None,
                fewshot_output_text=self.fewshot_output_text.__func__ if self.fewshot_output_text else None,
                fewshot_split=self.fewshot_split,
                num_fewshot=self.num_fewshot,
                sampler=self.sampler,
                fewshot_delimiter=self.fewshot_delimiter,
                answer_delimiter=self.answer_delimiter,
                n_samples=self.n_samples,
                data_kwargs=self.data_kwargs,
                batch_processing=self.batch_processing,
                )
        else:
            self.dataset = None

        if name is not None:
            self.name = name 
        #     self.name = self.__name__

        self.sample_agg_fn = getattr(self.sample_agg_fn, '__func__', self.sample_agg_fn)
        self.logging = getattr(self.logging, '__func__', self.logging)

        # self.preprocessor = get_preprocess_fn(preprocessor) if preprocessor else self.preprocessor
        self.preprocessor = getattr(self.preprocessor, '__func__', self.preprocessor)

        self.postprocessor = get_postprocess_fn(postprocessor) if postprocessor else self.postprocessor
        self.postprocessor = getattr(self.postprocessor, '__func__', self.postprocessor)

        self.system_role = system_role or self.system_role
        self.n_samples = n_samples or self.n_samples
        self.num_fewshot = num_fewshot or self.num_fewshot
        self.sampling_args = sampling_args or self.sampling_args
        
        # self.inference_fn = inference_fn
        # if isinstance(system_message, Prompt):
        #     self.system_message = system_message
        #     self.postprocessor = get_postprocess_fn(postprocessor)
        # else:
        #     self.system_message = system_message
        
        if isinstance(self.evaluation, Callable):
            self.evaluation = {"score": self.evaluation}
        elif isinstance(self.evaluation, str):
            if self.evaluation == "match":
                self.evaluation = {"match": match_fn}
            else:
                raise NotImplementedError

        if self.sampling_args is None:
            self.sampling_args = {}
        # self.sampling_args = sampling_args or {}
        # self.system_role = system_role

    def __len__(self):
        if self.dataset is None:
            return self.subtask_list[0].__len__()
        return len(self.dataset)

    def terminate(self):
        return True

    def preprocess(self, x, state=None):
        if self.preprocessor is not None:
            try:
                return self.preprocessor(x, state)
            except Exception as e:
                return self.preprocessor(x), state
        else:
            return x, state

    def postprocess(self, x, state=None):
        if self.postprocessor is not None:
            try:
                return self.postprocessor(x, state)
            except Exception as e:
                return self.postprocessor(x), state
        else:
            return x, state

    def build_message(self, x, state=None, system_role="system"):

        if "system_message" in state:
            system_message = state["system_message"]
        else:
            system_message = self.system_message

        if system_message in SYSTEM_MESSAGE:
            system_message = SYSTEM_MESSAGE[system_message]

        message = [{"role": "user", "content": x}]
        if system_message is not None:
            if system_role:
                message.insert(
                    0, 
                    {"role": system_role, "content": system_message}
                    )
            else:
                return [{"role": "user", "content": system_message+"\n\n"+x}]

        return message

    def extract_answer(self, prediction):
        if self.postprocessing is None:
            return prediction
        else:
            return self.postprocessing(prediction)

    def eval(self, prediction, ground_truth):
        return {
            eval_name: eval_fn(
                prediction,
                ground_truth
            ) for eval_name, eval_fn in self.evaluation.items()
        }


def create_task(
    name: str = None,
    subtask_list: list = None,
    dataset = None,
    preprocessor: callable = lambda x, y: (x, y),
    postprocessor: callable = lambda x, y: (x, y),
    inference_fn: callable = None,
    system_message: str = None,
    evaluation: Union[str, Dict[str, Callable]]="match",
    sampling_args: dict = {},
    logging: callable = None,
    ):
    return partial(Task, 
                name=name,
                subtask_list=subtask_list,
                dataset=dataset,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                inference_fn=inference_fn,
                system_message=system_message,
                evaluation=evaluation,
                sampling_args=sampling_args,
                logging=logging,
                )


if __name__ == "__main__":
    import concurrent.futures
    from openai import OpenAI
    from tqdm import tqdm
    from functools import partial
    from codethink.dataset.gsm8k import GSM8KDataset, GSM8KRoutingDataset

    def preprocess_PL_or_NL(x, state):
        current_step = state["current_step"]
        solve_with = state["step"][current_step-1]["output"]
        if solve_with == "programming language":
            state["system_message"] = "code"
        elif solve_with == "natural language":
            state["system_message"] = "cot"
        return x, state

    task_1 = Task(
        name="gsm8k_routing",
        dataset=GSM8KRoutingDataset,
        dataset_kwargs={"num_fewshot": 0},
        system_message="routing_selection_nl_first",
        )

    task_2 = Task(
        name="gsm8k_solve",
        dataset=GSM8KDataset,
        dataset_kwargs={"num_fewshot": 0},
        preprocessor=preprocess_PL_or_NL,
        )

    all_task = Task(
        name="gsm8k_pipeline",
        subtask_list=[
            task_1,
            task_2,
            ],
        )

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )

    kwargs = {"model": "Qwen/Qwen2.5-7B-Instruct"}

    def fetch_completion(messages, kwargs):
        try:
            response = client.chat.completions.create(
                messages=messages,
                temperature=0.8, top_p=0.95,
                **kwargs,
            )
            return {
                "response": [
                    response.choices[i].message.content
                    for i in range(0, len(response.choices))
                ],
                "input_token_len": response.usage.prompt_tokens,
                "output_token_len": response.usage.completion_tokens,
            }
        except Exception as e:
            print(f"Error fetching chat completion: {e}")
            return None

    # Use ThreadPoolExecutor for concurrent execution with a progress bar
    all_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                all_task.infer,
                idx,
                inference_fn=partial(
                    fetch_completion,
                    kwargs=kwargs
                    ),
                task_id="gsm8k_pipeline"
            )
            for idx in range(0,10)
        ]

        # Use tqdm to display a progress bar
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
            try:
                all_results.append(future.result())
                # print(f"Request {i} succeeded with response: {result}")
            except Exception as e:
                print(f"Request {i} failed with error: {e}")


            # self.get_answer_symbol = partial(
            #     extract_regex,
            #     fallback=fallback,
            #     regex=[
            #         re.compile("answer is (\\-?[0-9\\.\\,]*[0-9]+)"),
            #         re.compile("answer is (.*)."),
            #         ]
            #     )
