import os
from functools import partial
from typing import Union, Callable, Dict
from codethink._system_message import SYSTEM_MESSAGE

from transformers import AutoTokenizer
# Task
# Do initial inference of task
# Do parallel process?
# Route strings? i.e `next='task_name'`
# Merge and stuff

# Dataset
# Should fewshot be hardcoded?
# Should postprocessing change the dataset name? Or Task name?

# Build Instances, sample_id, task_id

def match_fn(x, y):
    try:
        return 1 if x == y else 0
    except Exception as e:
        return 0

class Task:
    NAME = None
    SUBTASK_LIST = None
    DATASET = None
    DATASET_KWARGS = {}
    PREPROCESSOR = None
    POSTPROCESSOR = None
    INFERENCE_FN = None
    SYSTEM_MESSAGE = None
    EVALUATION = None


    def __init__(self,
                 name,
                 subtask_list: list = None,
                 dataset = None,
                 dataset_kwargs: dict = {},
                 preprocessor: callable = lambda x, y: (x, y),
                 postprocessor: callable = lambda x, y: (x['response'][0], y),
                 inference_fn: callable = None,
                 tokenizer: str = None,
                 system_message: str = None,
                 evaluation: Union[str, Dict[str, Callable]]="match",
                 ):
        self.name = self.NAME or name
        self.subtask_list = self.SUBTASK_LIST or subtask_list
        self.dataset = self.DATASET or dataset
        self.dataset_kwargs = self.DATASET_KWARGS or dataset_kwargs
        if self.dataset is not None:
            self.dataset = self.dataset(**dataset_kwargs)
        
        self.preprocessor = self.PREPROCESSOR or preprocessor
        self.postprocessor = self.POSTPROCESSOR or postprocessor
        self.inference_fn = self.INFERENCE_FN or inference_fn
        self.system_message = self.SYSTEM_MESSAGE or system_message

        self.evaluation = self.EVALUATION or evaluation
        if isinstance(self.evaluation, Callable):
            self.evaluation = {"score": self.evaluation}
        elif isinstance(self.evaluation, str):
            if self.evaluation == "match":
                self.evaluation = {"match": match_fn}
            else:
                raise NotImplementedError

    def __len__(self):
        if self.dataset is None:
            return self.subtask_list[0].__len__()
        return len(self.dataset)

    def infer(self, idx, inference_fn=None, system_message=None):
        state = {
            "sample_id": idx,
            "current_step": 0,
            "step": []
            }

        if system_message is not None:
            self.system_message = system_message

        if self.subtask_list is None:
            output, _state = self.run_step(idx, state, inference_fn=inference_fn)
            state["step"].append(
                {"step_id": 0,
                 "task": self.name,
                 **_state}
            )
            return output, state

        for _id, task in enumerate(self.subtask_list):
            output, _state = task.run_step(idx,
                                           state=state,
                                           inference_fn=inference_fn
                                           )
            state["step"].append(
                {"step_id": _id,
                 "task": task.name,
                 **_state}
            )
            state["current_step"] += 1
        return output, state

    def preprocess(self, x, state=None):
        if self.preprocessor is not None:
            return self.preprocessor(x, state)
        else:
            return x, state

    def postprocess(self, x, state=None):
        if self.postprocessor is not None:
            return self.postprocessor(x, state)
        else:
            return x, state

    def build_message(self, x, state=None):
        message = [{"role": "user", "content": x}]

        if "system_message" in state:
            system_message = state["system_message"]
        else:
            system_message = self.system_message

        if system_message in SYSTEM_MESSAGE:
            system_message = SYSTEM_MESSAGE[system_message]

        if system_message is not None:
            message.insert(
                0, 
                {"role": "system", "content": system_message}
                )
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

    def run_step(self, idx, state=None, inference_fn=None):
        # State is what?
        # evals, inputs, outputs 
        # Accumulate states
        new_state = {}
        x, y = self.dataset.__getitem__(idx)
        new_state["raw_input"] = x
        new_state["ground_truth"] = y
        x, state = self.preprocess(x, state)
        x = self.build_message(x, state)
        new_state["full_input"] = x
        if inference_fn is None:
            o = self.inference_fn(x)
        else:
            o = inference_fn(x)
        new_state = {**new_state, **o}
        o, state = self.postprocess(o, state)
        new_state["output"] = o
        new_state["eval"] = self.eval(o, y)
        return o, new_state

def create_task(
    name: str = None,
    subtask_list: list = None,
    dataset = None,
    preprocessor: callable = lambda x, y: (x, y),
    postprocessor: callable = lambda x, y: (x['response'][0], y),
    inference_fn: callable = None,
    system_message: str = None,
    evaluation: Union[str, Dict[str, Callable]]="match",
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
