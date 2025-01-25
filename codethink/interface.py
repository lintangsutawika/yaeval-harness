import os
import re
import sys
import ray
import time
import logging
import itertools
import subprocess
import transformers
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

from typing import List
from functools import partial

from vllm import LLM, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)

def get_tokens(model_outputs: RequestOutput):

    all_output_tokens = []
    all_output_text = []
    input_tokens = len(list(model_outputs.prompt_token_ids))
    all_output_tokens = 0
    num = len(model_outputs.outputs)
    for output in model_outputs.outputs:

        output_text = output.text
        output_tokens = len(list(output.token_ids))

        if num == 1:
            return output_text, (input_tokens, output_tokens)

        all_output_text.append(output_text)
        all_output_tokens += output_tokens

    return all_output_text, (input_tokens, all_output_tokens)


class SolverInterface:
    def __init__(self,
                 model,
                 api_key="EMPTY",
                 api_base="http://localhost:8000/v1",
                 system_message,
                 repeat=1,
                 get_answer_symbol=None,
                 answer_expr='solution()',
                 fallback="[INVALID]",
                 verbose=False,
                 revision="main",
                 trust_remote_code=False,
                 use_system_role=False,
                 max_model_len=4096,
                 tensor_parallel_size=1,
                 data_parallel_size=1,
                 model_kwargs={},
                 **kwargs):


        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

        self.kwargs["model"] = model

    def fetch_completion(self, messages, kwargs):
        try:
            return self.client.chat.completions.create(
                messages=messages,
                **kwargs,
            )
        except Exception as e:
            print(f"Error fetching chat completion: {e}")
            return None

        self.answer_expr = answer_expr
        self.fallback = fallback
        self.verbose = verbose

        self.model_kwargs = model_kwargs
        if "stop" in self.model_kwargs:
            self.model_kwargs["stop"] = self.model_kwargs["stop"].replace("\\n", "\n")

        self.data_parallel_size = data_parallel_size
        if data_parallel_size <= 1:
            self.lm = LLM(
            # self.lm = TrajectoryControlLLM(
                model=self.model,
                revision=revision,
                max_model_len=max_model_len,
                # trust_remote_code=trust_remote_code,
                tensor_parallel_size=tensor_parallel_size
                )
        else:
            self.model_args = {
                "model": self.model,
                "revision": revision,
                "worker_use_ray": True,
                "max_model_len": max_model_len,
                # "trust_remote_code": trust_remote_code,
                "tensor_parallel_size":tensor_parallel_size
            }
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=trust_remote_code
            )
        self.history = []

    # Adapted from 
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/vllm_causallms.py
    @ray.remote
    def run_inference_one_model(
        model_args: dict,
        sampling_params,
        requests: List[List[int]],
        # lora_request: LoRARequest,
        rank: int,
    ):
        del os.environ['CUDA_VISIBLE_DEVICES']
        llm = LLM(**model_args)
        return llm.generate(
            prompts=requests,
            sampling_params=sampling_params,
            use_tqdm=True if rank == 0 else False,
            # lora_request=lora_request,
        )

    def generate(self, message, sampling_params):
        output = self.lm.generate(message, sampling_params, use_tqdm=True if self.verbose else False)
        return output

    def run(self, prompt, time_out: float = 10, temperature: float = 0, top_p: float = 1, max_tokens: int = 512, repeat: int = 1, seed: int = None):
        if isinstance(prompt, str):
            prompt = [prompt]
        try:
            if self.use_system_role:
                message =[[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': p}] for p in prompt]
            else:
                message =[[{'role': 'user', 'content': self.system_message+"\n\n"+p}] for p in prompt]
            message = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            message = self.system_message+"\n\n"+prompt
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=repeat, seed=seed, **self.model_kwargs)
        # sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=repeat, seed=seed, stop=self.stop, include_stop_str_in_output=True)
        # sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=repeat, seed=seed, stop=["```\n", "``` \n"], include_stop_str_in_output=True)
        start_time = time.time()

        if self.data_parallel_size > 1:

            from more_itertools import distribute
            from typing import Iterable
            import os

            def undistribute(iterable):
                return [
                    x
                    for x in itertools.chain.from_iterable(
                        itertools.zip_longest(*[list(x) for x in iterable])
                    )
                    if x is not None
                ]

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, message)]
            inputs = (
                (self.model_args, sampling_params, req, rank)
                for rank, req in enumerate(requests)
            )
            object_refs = [self.run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            model_generations = undistribute(results)
        else:
            model_generations = self.generate(message, sampling_params)

        all_answers = []
        all_outputs = []
        for model_output in model_generations:
            output, (input_len, output_len) = get_tokens(model_output)

            answer_dict = {}
            if isinstance(output, str):
                output = [output]

            for _output in output:
                if self.verbose:
                    print(_output)
                self.history.append(_output)
                # all_outputs.append(_output)

                # Check if the _output is a program
                code = self.process_generation_to_code(_output)
                if code:
                    def _generate_code(code, answer_expr):
                        return "\n".join(code)+f"\nans = 'ans='+str({answer_expr})\nprint(ans)"
                    # Generate code snippet that will be executed in a different process
                    code_snippet = _generate_code(code, self.answer_expr)
                    try:
                        subprocess_result = subprocess.run([sys.executable, "-c", code_snippet], timeout=time_out, text=True, capture_output=True)
                        exec_result = subprocess_result.stdout.split("ans=")[-1].strip()
                    except Exception as e:
                        print(e)
                        exec_result = ""

                    if exec_result in answer_dict:
                        answer_dict[exec_result] += 1
                    else:
                        answer_dict[exec_result] = 1
                else:
                    if self.get_answer_symbol is not None:
                        match = self.get_answer_symbol(_output)
                        if match in answer_dict:
                            answer_dict[match] += 1
                        else:
                            answer_dict[match] = 1

            if repeat == 1:
                result = list(answer_dict.keys())[0]
            else:
                counts = list(answer_dict.values())
                max_idx = counts.index(max(counts))
                result = list(answer_dict.keys())[max_idx]

            duration = time.time() - start_time
            output_dict = {
                "input_len": input_len,
                "output_len": output_len,
                "duration": duration,
                "system_output": output,
            }
            all_answers.append(result)
            all_outputs.append(output_dict)
        return all_answers, all_outputs

if __name__ == "__main__":

    import re
    from tqdm import tqdm
    from datasets import load_dataset

    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    model_str = "meta-llama/Llama-3.1-8B-Instruct"
    system_message = '''\
Write a function to solve a given problem by the user. Only write the program. Do not use `print`.
The function must be named solution() and return `value` where value is only a number without any signs like '$' or '%'.\
'''
    model = SolverInterface(
        system_message=system_message,
        model=model_str,
        get_answer_expr='solution()',
        data_parallel_size=2,
        )

    all_scores = []

    requests = []
    task_samples = []
    for sample in tqdm(gsm8k_test):
        requests.append(sample["question"])
        task_samples.append(sample)

    output, _ = model.run(requests, temperature=0.1)

    for ans, sample in tqdm(zip(output, task_samples)):
        question = sample["question"]
        gt = sample["answer"]
        gt = gt.split("#### ")[-1]
        gt = float(re.findall(r'\d+', gt)[0])
        try:
            ans = float(ans)
            score = 1 if abs(ans - gt) < 1e-3 else 0
        except Exception as e:
            ans = ''
            score = 0
        all_scores.append(score)

    print(f'Accuracy - {sum(all_scores) / len(all_scores)}')
