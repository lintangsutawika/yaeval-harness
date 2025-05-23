import os
import itertools
import numpy as np

from dataclasses import dataclass
from functools import partial
from typing import Union, Callable, Dict, List
from yeval.prompt import YevalPrompt, get_message, get_prompt
from yeval.response import get_postprocess_fn

from transformers import AutoTokenizer

def match_fn(x, y):
    try:
        return 1 if x == y else 0
    except Exception as e:
        return 0

class YevalTask:

    name: str = None
    subtask_list: list = None
    preprocessor: Union[str, Callable] = None
    postprocessor: Union[str, Callable] = None
    inference_fn: callable = None
    tokenizer: str = None
    prompt_message: Union[str, YevalPrompt] = None
    system_message: Union[str, Callable] = None
    user_message: Union[str, Callable] = None
    evaluation: Union[str, Dict[str, Callable]]="match"
    sampling_args: dict =None
    system_role: str = "assistant"
    logging: callable = None
    sample_agg_fn: Union[dict, Callable] = np.mean
    dataset = None
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
    loop_exit: Callable=None
    loop_max: int=1
    eval_at_k: bool=False
    subtask_fn: Union[str, Callable]=None
    aux_keys: List[str]=None

    @staticmethod
    def _input_text(self, x):
        return self.input_text(x)


    def __init__(
        self,
        name: str = None,
        data_path: str = None,
        subtask_list: list = None,
        preprocessor: Union[str, Callable] = None,
        postprocessor: Union[str, Callable] = None,
        prompt_message:Union[str, YevalPrompt] = None,
        system_message: Union[str, Callable] = None,
        user_message: Union[str, Callable] = None,
        system_role: str = None,
        sampling_args: dict = None,
        sample_agg_fn: Union[dict, Callable] = None,
        num_fewshot: Union[int, float] = None,
        n_samples: Union[int, float] = None,
        dataset = None,
        evaluation: Union[str, Dict[str, Callable]] = None,
        subtask_fn: Union[str, Callable] = None,
        data_kwargs: dict = None,
        aux_keys: List[str] = None,
        preprocessing: Union[str, Callable] = None,
        test_split: str = None,
        **kwargs,
        ):

        self.data_path = self.data_path or data_path
        self.data_kwargs = data_kwargs or self.data_kwargs
        self.preprocessing = preprocessing or self.preprocessing
        if dataset is not None:
            self.dataset = dataset
        else:
            if self.data_path is not None:
                from yeval.task import YevalDataset
                self.dataset = YevalDataset(
                    data_path=self.data_path,
                    data_name=self.data_name,
                    input_text=getattr(self.input_text, '__func__', self.input_text),
                    output_text=getattr(self.output_text, '__func__', self.output_text),
                    preprocessing=self.preprocessing.__func__ if self.preprocessing else None,
                    test_split=test_split or self.test_split,
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
                    aux_keys=aux_keys or self.aux_keys,
                    )
            else:
                self.dataset = None

        self.sample_agg_fn = sample_agg_fn or self.sample_agg_fn
        if isinstance(self.sample_agg_fn, Callable):
            self.sample_agg_fn = getattr(self.sample_agg_fn, '__func__', self.sample_agg_fn)
        elif isinstance(self.evaluation, dict):
            for key, value in self.sample_agg_fn.items():
                self.sample_agg_fn[key] = getattr(value, '__func__', value)

        self.logging = getattr(self.logging, '__func__', self.logging)

        if system_role is False:
            self.system_role = False
        else:
            self.system_role = system_role or self.system_role
        self.n_samples = n_samples or self.n_samples
        self.num_fewshot = num_fewshot or self.num_fewshot
        self.sampling_args = sampling_args or self.sampling_args
        
        if preprocessor is not None:
            self.preprocessor = preprocessor
        self.preprocessor = getattr(self.preprocessor, '__func__', self.preprocessor)

        _system_message = None
        _user_message = None
        _postprocessor = None

        # Overriding System message
        if system_message is not None:
            self.system_message = system_message

        if prompt_message is not None:
            self.prompt_message = prompt_message
        if isinstance(self.prompt_message, str):
            _system_message, _user_message, _postprocessor = get_prompt(self.prompt_message)
        elif isinstance(self.prompt_message, YevalPrompt):
            _system_message, _user_message, _postprocessor = self.prompt_message()

        self.system_message = _system_message or system_message or self.system_message
        self.user_message = _user_message or user_message or self.user_message

        if _postprocessor is None:
            # postprocessor can be overwritten by the system_message
            self.postprocessor = get_postprocess_fn(postprocessor or self.postprocessor)
            self.postprocessor = getattr(self.postprocessor, '__func__', self.postprocessor)
        else:
            self.postprocessor = _postprocessor

        if _system_message is not None:
            self.system_message = _system_message
        
        if _user_message is not None:
            self.user_message = _user_message

        self.evaluation = evaluation or self.evaluation
        if isinstance(self.evaluation, Callable):
            self.evaluation = {"score": self.evaluation}
        elif isinstance(self.evaluation, str):
            if self.evaluation == "match":
                self.evaluation = {"match": match_fn}
            else:
                raise NotImplementedError

        if self.sampling_args is None:
            self.sampling_args = {}

        self.terminate = False
        self.loop_exit = getattr(self.loop_exit, '__func__', self.loop_exit)

        self.subtask_list = subtask_list or self.subtask_list
        if self.subtask_list is not None:
            self.subtask_list = [
                task(
                    **{
                        key: getattr(self, key)
                        for key in dir(self)
                        if not key.startswith("__") and not callable(getattr(self, key)) and key != "subtask_list"
                    }
                )
                for task in self.subtask_list
            ]

        self.subtask_fn = subtask_fn or getattr(self.subtask_fn, '__func__', self.subtask_fn)
        self.name = name or type(self).__name__

    def __len__(self):
        if self.dataset is None:
            return self.subtask_list[0].__len__()
        return len(self.dataset)

    def next_subtask(self, state=None, subtask_iter=None):

        assert len(self.subtask_list) > 0, "No subtask list found"
        if self.subtask_fn is None:
            try:
                next_task = next(subtask_iter)
            except StopIteration:
                return None, True
            return next_task, False
        else:
            next_task, exit_iter = self.subtask_fn(state, self.subtask_list)
            if next_task is None:
                return None, True
            return next_task, exit_iter

    def check_termination(self, x, state, fn=None):
        fn = fn or self.loop_exit
        current_loop = state["current_loop"]
        if current_loop == (self.loop_max-1):
            self.terminate = True
        else:
            current_loop += 1
            if fn is not None:
                try:
                    self.terminate = fn(x, state)
                except Exception as e:
                    self.terminate = fn(x)

    def preprocess(self, x, state=None, fn=None):
        fn = fn or self.preprocessor
        if fn is not None:
            try:
                return fn(x, state)
            except Exception as e:
                return fn(x), state
        else:
            return x, state

    def postprocess(self, x, state=None, fn=None):
        fn = fn or self.postprocessor
        if fn is not None:
            try:
                return fn(x, state)
            except Exception as e:
                return fn(x), state
        else:
            return x, state

    def build_message(self, x, state=None, system_message=None, user_message=None, chat=True):
        if system_message is not None:
            system_message = system_message
        elif "system_message" in state:
            system_message = state["system_message"]
        else:
            system_message = self.system_message

        if isinstance(system_message, Callable):
            system_message = system_message(x)

        if user_message is not None:
            user_message = user_message
        elif "user_message" in state:
            user_message = state["user_message"]
        else:
            user_message = self.user_message

        if user_message is None:
            user_message = x
        elif isinstance(user_message, Callable):
            try:
                user_message = user_message(x)
            except:
                user_message = user_message.__func__(x)
        elif isinstance(user_message, str):
            user_message = user_message + "\n" + x

        if chat:
            message = [{"role": "user", "content": user_message}]
            if system_message:
                if self.system_role:
                    message.insert(0, {"role": self.system_role, "content": system_message})
                else:
                    message = [{"role": "user", "content": f"{system_message}\n\n{user_message}"}]
        else:
            message = f"{system_message}\n\n{user_message}" if system_message else user_message

        return message

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
    from yeval.dataset.gsm8k import GSM8KDataset, GSM8KRoutingDataset

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
