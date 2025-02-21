import os
import json
import time
import logging
import datetime
import jsonlines
import concurrent.futures

from openai import OpenAI
from tqdm import tqdm

from functools import partial
from typing import List, Tuple
from torch.utils.data import DataLoader

from vllm import SamplingParams
from codethink.utils import zeno_upload, check_api_health

logger = logging.getLogger(__name__)

class EvaluateSystem:
    def __init__(self,
                 model,
                 api_key="EMPTY",
                 api_base="http://localhost:8000/v1",
                 output_path=None,
                 run_args=None,
                 use_run_name=True,
                 verbose=False,
                 sampling_args=None,
                 system_message=None,
                 max_requests=128,
                 **kwargs,
                 ):

        self.model = model
        self.output_path = output_path
        self.run_args = run_args
        self.use_run_name = use_run_name
        self.verbose = verbose
        self.max_requests = max_requests

        self.sampling_args = sampling_args or {}
        self.sampling_args["model"] = model

        while check_api_health(
            api_base.split("/v1")[0]+"/health"
            ) is False:

            logger.info("API is not available, retrying...")
            time.sleep(5)

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

        self.system_message = system_message

    def fetch_completion(self, messages, sampling_args=None):
        if sampling_args is not None:
            sampling_args = {**self.sampling_args, **sampling_args}
        else:
            sampling_args = self.sampling_args

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                **sampling_args,
            )
            return {
                "response": [
                    response.choices[i].message.content
                    for i in range(0, len(response.choices))
                ],
                "input_len": response.usage.prompt_tokens,
                "output_len": response.usage.completion_tokens,
            }
        except Exception as e:
            print(f"Error fetching chat completion: {e}")
            return None

    def run(self, task, sampling_args=None, run_name=None, n_samples=None):

        if run_name is None:
            current_time = datetime.datetime.now()
            self.run_name = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        else:
            self.run_name = run_name

        result_dict = {
            "n_samples": 0,
            "duration": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        data_dict = {
            "idx": [],
            "answer": [],
            "system_output": [],
            "ground_truth": [],
            "user_input": [],
            "score": [],
            "duration": [],
            "input_tokens": [],
            "output_tokens": [],
            "total_tokens": [],
        }
        output_json = []
        idx = 0

        user_input = []
        ground_truth = []
        # Use ThreadPoolExecutor for concurrent 
        # execution with a progress bar

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

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                        task.infer,
                        idx,
                        inference_fn=self.fetch_completion,
                        sampling_args=sampling_args,
                        system_message=self.system_message,
                    ) for idx in range(n_samples)
                ]

            all_results = []
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
                all_results.append(future.result())
                #try:
                #    all_results.append(future.result())
                #except Exception as e:
                #    print(f"Request {i} failed with error: {e}")
                #    pass

        # all_results = [x for x in sorted(all_results, key=lambda x: x["sample_id"])]
        for ans, steps in tqdm(all_results):
            output_dict = steps["step"][-1]
            inp = output_dict["full_input"]
            gt = output_dict["ground_truth"]
            score_dict = output_dict['eval']
            score_string = ", ".join([
                f"{metric}: {score}" for metric,score in score_dict.items()
                ])

            if self.verbose:
                logger.info(f"\nId: {idx}, {score_string}, Prediction: {ans}, Ground Truth: {gt}")
            result_dict["n_samples"] += 1
            for metric, score in score_dict.items():
                if metric not in result_dict:
                    result_dict[metric] = score
                else:
                    result_dict[metric] += score

            # result_dict["duration"] += output_dict["duration"]
            result_dict["input_tokens"] += output_dict["input_len"]
            result_dict["output_tokens"] += output_dict["output_len"]
            result_dict["total_tokens"] += (output_dict["input_len"] + output_dict["output_len"])
            output_json.append(
                {
                    "idx": idx,
                    **score_dict,
                    "ground_truth": gt,
                    "answer": ans,
                    "user_input": inp,
                    **output_dict,
                    **steps,
                }
            )

            # data_dict["idx"].append(int(idx))
            # data_dict["answer"].append(ans)
            # data_dict["system_output"].append(output_dict["system_output"])
            # data_dict["ground_truth"].append(gt)
            # data_dict["user_input"].append(user_input)
            # data_dict["score"].append(score)
            # data_dict["duration"].append(output_dict["duration"])
            # data_dict["input_tokens"].append(output_dict["input_len"])
            # data_dict["output_tokens"].append(output_dict["output_len"])
            # data_dict["total_tokens"].append(output_dict["input_len"] + output_dict["output_len"])
            idx += 1
        # zeno_upload(self.run_name, data_dict)

        # result_dict = {**self.run_args, **result_dict}
        for metric, score in score_dict.items():
            result_dict[f"avg_{metric}"] = result_dict[metric]/result_dict["n_samples"]
        # result_dict["avg_duration"] = result_dict["duration"]/result_dict["n_samples"]
        result_dict["avg_input_tokens"] = result_dict["input_tokens"]/result_dict["n_samples"]
        result_dict["avg_output_tokens"] = result_dict["output_tokens"]/result_dict["n_samples"]
        result_dict["avg_total_tokens"] = result_dict["total_tokens"]/result_dict["n_samples"]
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

