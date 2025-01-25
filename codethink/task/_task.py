import os

from codethink import SYSTEM_MESSAGE

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

class Task:
    def __init__(self,
                 name,
                 subtask_list: list = None,
                 dataset = None,
                 dataset_kwargs: dict = {},
                 preprocessor: callable = None,
                 postprocessor: callable = None,
                 inference_fn: callable = None,
                 tokenizer: str = None,
                 system_message: str = None,
                 evaluation: callable = None,
                 ):
        self.name = name
        self.subtask_list = subtask_list
        if dataset is not None:
            self.dataset = dataset(**dataset_kwargs)
        else:
            self.dataset = None
        self.preprocessor = preprocessor

        if postprocessor is None:
            self.postprocessor = lambda x, y: x[0]
        else:
            self.postprocessor = postprocessor

        if inference_fn is None:
            self.inference_fn = lambda x: x
        else:
            self.inference_fn = inference_fn

        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            [task.set_tokenizer(tokenizer) for task in self.subtask_list]
        else:
            self.tokenizer = None

        if system_message is not None:
            self.system_message = system_message
        else:
            self.system_message = None

        if evaluation is not None:
            self.eval = evaluation
        else:
            self.eval = lambda x, y: -1

    def set_tokenizer(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def run(self, idx, task_id=None, inference_fn=None):
        state = {
            "task_id": task_id,
            "sample_id": idx,
            "current_step": 0,
            "step": []
            }

        if self.subtask_list is None:
            output, _state = self.run_task(idx, state)
            state["step"].append({"step_id": 0, "task": self.name, **_state})
            return output, state

        for _id, task in enumerate(self.subtask_list):
            output, _state = task.run_task(idx,
                                          state=state,
                                          inference_fn=inference_fn
                                          )
            state["step"].append({"step_id": _id, "task": task.name, **_state})
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
        elif self.tokenizer is not None:
            message = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                )
        return message

    def run_task(self, idx, state=None, inference_fn=None):
        # State is what?
        # evals, inputs, outputs 
        # Accumulate states
        new_state = {}
        x, y = self.dataset.__getitem__(idx)
        new_state["raw_input"] = x
        new_state["groud_truth"] = y
        x, state = self.preprocess(x, state)
        x = self.build_message(x, state)
        new_state["full_input"] = x
        if inference_fn is None:
            o = self.inference_fn(x)
        else:
            o = inference_fn(x)

        new_state["raw_output"] = o
        o = self.postprocess(o, state)
        new_state["output"] = o
        new_state["eval"] = self.eval(o, y)
        return o, new_state

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
        # tokenizer="meta-llama/Llama-3.1-8B-Instruct"
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
                **kwargs,
            )
            return [response.choices[i].message.content for i in range(0, len(response.choices))]
        except Exception as e:
            print(f"Error fetching chat completion: {e}")
            return None

    # Use ThreadPoolExecutor for concurrent execution with a progress bar
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(all_task.run, idx, inference_fn=partial(fetch_completion, kwargs=kwargs), task_id="gsm8k_pipeline")
            for idx in range(0,10)
        ]

        # Use tqdm to display a progress bar
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
            try:
                result = future.result()
                # print(f"Request {i} succeeded with response: {result}")
            except Exception as e:
                print(f"Request {i} failed with error: {e}")

