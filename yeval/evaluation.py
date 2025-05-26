import os
import json
import time
import logging
import datetime
import jsonlines
import concurrent.futures
import asyncio
from tqdm.asyncio import tqdm

from openai import OpenAI, AsyncOpenAI
# from tqdm import tqdm

from functools import partial
from typing import List, Tuple
from torch.utils.data import DataLoader

from yeval.prompt import get_prompt
from yeval.response import get_postprocess_fn
from yeval.utils import check_api_health
from yeval.utils.api_postprocess import vllm_postprocess, openai_completion_postprocess

logger = logging.getLogger(__name__)

import time
import sys
import traceback
from tqdm.asyncio import tqdm
from dataclasses import dataclass, field
from typing import Optional, Union

import aiohttp

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    language: Optional[str] = None

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""

async def async_request_openai_completions(
    api_base,
    api_key,
    prompt,
    pbar,
    **sampling_args,
) -> RequestFuncOutput:
    api_url = api_base + "completions"
    assert api_url.endswith(("completions", "profile")), (
        "OpenAI Completions API URL must end with 'completions' or 'profile'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "prompt": prompt,
            **sampling_args
        }

        headers = {"Authorization": f"Bearer {api_key}"}

        output = RequestFuncOutput()
        # output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!"
                        )
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

class EvaluateSystem:
    def __init__(self,
                 model,
                 api_key="EMPTY",
                 api_base="http://localhost:8000/v1",
                 api_process=None,
                 output_path=None,
                 run_args=None,
                 use_run_name=True,
                 verbose=False,
                 sampling_args=None,
                 prompt_message=None,
                 system_message=None,
                 user_message=None,
                 postprocessor=None,
                 system_role="assistant",
                 max_rps=1000,
                 chat_completion=True,
                 max_new_tokens=512,
                 **kwargs,
                 ):

        self.model = model
        self.output_path = output_path
        self.run_args = run_args
        self.use_run_name = use_run_name
        self.verbose = verbose
        self.max_rps = max_rps

        self.sampling_args = sampling_args or {}
        self.sampling_args["model"] = model

        while True:
            check_1 = check_api_health(api_base.split("/v1")[0]+"/health")
            check_2 = check_api_health(api_base.split("/v1")[0])
            if check_1 or check_2:
                break

            logger.info("API is not available, retrying...")
            time.sleep(5)

        # self.client = OpenAI(
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.api_base = api_base
        self.api_key = api_key
        if api_process is None:
            if chat_completion:
                self.api_process = vllm_postprocess
            else:
                self.api_process = openai_completion_postprocess

        self.system_role = system_role or "assistant"
        self.system_message, self.user_message, self.postprocessor = get_prompt(prompt_message)
        
        self.system_message = system_message or self.system_message
        self.user_message = user_message or self.user_message
        # postprocessor can be overwritten by the system_message
        self.postprocessor = get_postprocess_fn(postprocessor or self.postprocessor)
        self.postprocessor = getattr(self.postprocessor, '__func__', self.postprocessor)

        self.chat_completion = chat_completion
        self.max_new_tokens = max_new_tokens

    async def fetch_chat_completion(self, messages, sampling_args=None):
        if sampling_args is not None:
            sampling_args = {**sampling_args, **self.sampling_args}
        else:
            sampling_args = self.sampling_args

        try:
            response = await self.client.chat.completions.create(
                messages=messages,
                **sampling_args,
            )
            return self.api_process(response)
            # return response
        # except asyncio.CancelledError:
        #     return None
        except Exception as e:
            print(f"Error fetching chat completion: {e}")
            return [""], {}

    async def fetch_completion(self, messages, sampling_args=None):
        if sampling_args is not None:
            sampling_args = {**sampling_args, **self.sampling_args}
        else:
            sampling_args = self.sampling_args

        if "max_tokens" not in sampling_args:
            # sampling_args["max_tokens"] = self.max_new_tokens
            sampling_args["max_tokens"] = 512

        try:
            response = await self.client.completions.create(
                prompt=messages,
                **sampling_args,
            )
            return self.api_process(response)

            # response = await async_request_openai_completions(
            #     self.api_base,
            #     self.api_key,
            #     prompt=messages,
            #     **sampling_args,
            # )
            # return [response.generated_text], {
            #     "latency": response.latency,
            #     "output_tokens": response.output_tokens,
            #     "ttft": response.ttft,
            #     "itl": response.itl,
            #     "tpot": response.tpot,
            #     "prompt_len": response.prompt_len,
            # }
            # return response
        # except asyncio.CancelledError:
        #     return None
        except Exception as e:
            print(f"Error fetching completion: {e}")
            return [""], {}

    async def run(self, task, sampling_args=None, run_name=None, n_samples=None):

        if run_name is None:
            current_time = datetime.datetime.now()
            self.run_name = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        else:
            self.run_name = run_name

        #if system_message is not None:
        #    self.system_message = system_message

        if sampling_args is not None:
            self.sampling_args = {**self.sampling_args, **sampling_args}

        result_dict = {
            "n_samples": 0,
        }

        output_json = []
        idx = 0

        user_input = []
        ground_truth = []

        n_samples = n_samples or task.__len__()

        def chunk_len(num, max_request):
            ranges = []
            n = num//max_request
            for i in range(n):
                ranges.append((0+i*max_request,max_request+i*max_request))

            modulo = num%max_request
            if modulo > 0:
                ranges.append((n*max_request, n*max_request+modulo))

            return ranges

        n_ranges = chunk_len(n_samples, self.max_rps)

        # all_results = []
        # with tqdm(total=n_samples) as pbar:
        #     for n_range in n_ranges:
        #         all_requests = [
        #             self.infer(
        #             task,
        #             idx,
        #             ) for idx in range(*n_range)
        #         ]
        #         for completion in asyncio.as_completed(all_requests):
        #             result = await completion
        #             if result is not None:
        #                 all_results.append(result)
        #             pbar.update(1)

        all_results = []
        # semaphore = asyncio.Semaphore(self.max_rps)  # Limit concurrent requests
        tasks: list[asyncio.Task] = []

        async def get_dataset_item(
            task,
            n_samples
        ):
            for idx in range(n_samples):
                x, y = task.dataset.__getitem__(idx)
                yield idx, x, y

        async def get_request(
            input_requests: list,
        # ) -> AsyncGenerator:
        ):
            input_requests: Iterable[SampleRequest] = iter(input_requests)
            for request in input_requests:
                yield request

        with tqdm(total=n_samples) as pbar:
            async for (idx, x, y) in get_dataset_item(task, n_samples):
                tasks.append(
                    asyncio.create_task(
                        # self.run_step(task, idx, x=x)
                        self.client.completions.create(
                            # prompt=request,
                            prompt="f{idx}"+x,
                            model="Qwen/Qwen3-4B",
                            max_tokens=512,
                        )
                    )
                )

            # all_results = await asyncio.gather(*tasks)
            for completed_task in asyncio.as_completed(tasks):
                result = await completed_task
                all_results.append(result)
                pbar.update(1)

        for ans, steps in tqdm(all_results):
            output_dict = steps["step"][-1]
            inp = output_dict["full_input"]
            gt = output_dict["ground_truth"]
            score_dict = output_dict["eval"]
            score_string = ", ".join([
                f"{metric}: {score}" for metric,score in score_dict.items()
                ])

            if self.verbose:
                logger.info(f"\nId: {idx}, {score_string}, Prediction: {ans}, Ground Truth: {gt}")
            result_dict["n_samples"] += 1
            for metric, score in score_dict.items():
                if metric not in result_dict:
                    result_dict[metric] = score.copy()
                else:
                    result_dict[metric] += score.copy()

            if "log" in output_dict:
                for key, value in output_dict["log"].items():
                    if key in result_dict:
                        result_dict[key] += value
                    else:
                        result_dict[key] = value
            output_json.append(
                {
                    "idx": idx,
                    **score_dict,
                    "ground_truth": gt,
                    "answer": ans,
                    **(output_dict["log"] if "log" in output_dict else {}),
                    **{k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v)) for k, v in steps.items()},
                }
            )

            idx += 1

        result_keys = list(result_dict.keys())
        for key in result_keys:
            if key == "n_samples":
                continue
            try:
                result_dict[f"avg_{key}"] = result_dict[key]/result_dict["n_samples"]
            except:
                result_dict[f"avg_{key}"] = -1
            result_dict.pop(key)
        
        logger.warning(f"{self.run_name} complete")
        logger.warning(
            ",".join(
                [f"{metric}: {result_dict[f'avg_{metric}']}" for metric in score_dict.keys()]
                )
            )

        if self.output_path is not None:
            if self.use_run_name:
                run_path = os.path.join(self.output_path, self.run_name)
            else:
                run_path = os.path.join(self.output_path)
            os.makedirs(run_path, exist_ok=True)
            result_file = os.path.join(run_path, "result.json")
            with open(result_file, 'w', encoding='utf-8') as file:
                json.dump(result_dict, file, ensure_ascii=False, indent=4)

            try:
                output_file = os.path.join(run_path, "output.jsonl")
                with jsonlines.open(output_file, "w") as file:
                    file.write_all(output_json)
            except Exception as e:
                print("Error:", e)

        return 0

    async def run_step(self, task, idx, state=None, sampling_args=None, x=None, pbar=None):
        sampling_args = sampling_args or {}
        new_state = {}
        # x, y = task.dataset.__getitem__(idx)
        # new_state["aux"] = task.dataset.__getaux__(idx)
        # new_state["ground_truth"] = y
        # x, state = task.preprocess(x, state)
        # message_args = {}
        # if self.system_message is not None:
        #     message_args["system_message"] = self.system_message
        # if self.user_message is not None:
        #     message_args["user_message"] = self.user_message
        # x = task.build_message(x, state, **message_args, chat=self.chat_completion)
        # new_state["full_input"] = x
        if self.chat_completion:
            o, _state = await self.fetch_chat_completion(x, task.sampling_args)
        else:
            o, _state = await self.fetch_completion(x, task.sampling_args)
        # task.check_termination(o[0], state)
        # new_state["completion"] = o
        # if task.logging:
        #     new_state["log"] = task.logging(_state)
        #     # new_state["log"] = {}
        # # new_state = {**new_state, **_state}
        # if isinstance(o, list):
        #     o = [task.postprocess(_o, {**state, **new_state}, fn=self.postprocessor)[0] for _o in o]
        #     if task.eval_at_k:
        #         sample_score = [task.eval(o, y)]
        #     else:
        #         sample_score = [task.eval(_o, y) for _o in o]
        #     new_state["eval"] = {}
        #     for score in sample_score:
        #         for metric_name, metric_score in score.items():
        #             if metric_name in new_state["eval"]:
        #                 new_state["eval"][metric_name].append(metric_score)
        #             else:
        #                 new_state["eval"][metric_name] = [metric_score]
        #     if task.sample_agg_fn:
        #         if isinstance(task.sample_agg_fn, dict):
        #             new_state["eval"] = {
        #                 k: task.sample_agg_fn[k](
        #                     new_state["eval"][k]
        #                     ) for k in new_state["eval"].keys()
        #                 }
        #         else:
        #             new_state["eval"] = {
        #                 k: task.sample_agg_fn(
        #                     new_state["eval"][k]
        #                     ) for k in new_state["eval"].keys()
        #                 }
        # else:
        #     o, state = task.postprocess(o, {**state, **new_state}, fn=self.postprocessor)
        #     new_state["eval"] = self.eval(o, y)

        # new_state["output"] = o

        if pbar is not None:
            pbar.update(1)
        return o, new_state

    async def infer(self, task, idx, state=None):
        if state is None:
            state = {
                "sample_id": idx,
                "current_step": 0,
                "task_step": 0,
                "step": []
                }

        if task.subtask_list is None:
            state["current_loop"] = 0
            while True:
                output, _state = await self.run_step(
                                                task,
                                                idx,
                                                state,
                                                )
                state["step"].append(
                    {
                        "step_id": 0,
                        "task": task.name,
                        **_state
                        }
                )
                state["current_step"] += 1
                state["current_loop"] += 1
                if task.terminate:
                    break
            state["task_step"] += 1
            return output, state

        _id = state["task_step"]
        state["current_loop"] = 0
        subtask_iter = iter(task.subtask_list)
        _task, exit_loop = task.next_subtask(state=state, subtask_iter=subtask_iter)
        while True:
            if _task is not None:
                if exit_loop:
                    _id = state["task_step"]
                    state["task_step"] += 1
                    output, _state = await self.run_step(
                                                _task,
                                                idx,
                                                state=state,
                                                )
                    state["step"].append(
                        {
                            "step_id": _id,
                            "task": _task.name,
                            **_state
                            }
                    )
                else:
                    output, _state = await self.infer(_task, idx, state=state)

            if exit_loop:
                break

            _task, exit_loop = task.next_subtask(state=state, subtask_iter=subtask_iter)

        return output, state
