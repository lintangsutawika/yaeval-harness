import re
import logging
import torch
import transformers

import pal
from flops_profiler.profiler import FlopsProfiler

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

class HFProgramInterface(pal.interface.ProgramChatInterface):
    def __init__(self,
                 *args,
                revision="main",
                device=None,
                model_kwargs={},
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=False,
                **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.lm = transformers.pipeline(
            "text-generation",
            model=self.model,
            revision=revision,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
        )
        self.lm.generation_config.pad_token_id = self.lm.tokenizer.eos_token_id
        self.profile = FlopsProfiler(self.lm.model)

    def generate(self, prompt: str, temperature: float = 0.1, top_p: float = 1, max_tokens: int = 512):
        message =[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': prompt}]
        # message = self.lm.tokenizer.apply_chat_template(
        #             message,
        #             tokenize=False,
        #             return_dict=True,
        #             add_generation_prompt=True
        #             )
        output = self.lm(message, temperature=temperature, top_p=top_p, max_new_tokens=max_tokens)
        program = output[0]["generated_text"][-1]['content']
        if self.verbose:
            print(program)
        self.history.append(program)
        return self.process_generation_to_code(program)

    def run(self, prompt: str, time_out: float = 10, temperature: float = 0, top_p: float = 1, max_tokens: int = 512, return_generation=False):
        self.profile.start_profile()
        code = self.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        with pal.interface.timeout(time_out):
            try:
                exec_result = self.execute(code)
            except Exception as e:
                print(e)
        self.profile.stop_profile()
        flops = self.profile.get_total_flops()
        # macs = self.profile.get_total_macs()
        # params = self.profile.get_total_params()
        # if print_profile:
        #     self.profile.print_model_profile(profile_step=profile_step)
        self.profile.end_profile()
        if return_generation:
            return exec_result, flops, code
        return exec_result, flops


class HFNatLangInterface:
    def __init__(self,
                 model,
                 system_message,
                 repeat=1,
                 get_answer_symbol=None,
                 fallback="[INVALID]",
                 verbose=False,
                 revision="main",
                 device=None,
                 model_kwargs={},
                 device_map="auto",
                 torch_dtype="auto",
                 trust_remote_code=False,
                 **kwargs):

        self.system_message = system_message
        self.repeat = repeat
        # self.get_answer_symbol = get_answer_symbol
        self.get_answer_symbol = re.compile(get_answer_symbol)
        self.fallback = fallback
        self.verbose = verbose

        # self.lm = transformers.pipeline(
        #     "text-generation",
        #     model=LLM(model=model),
        #     revision=revision,
        #     device=device,
        #     device_map=device_map,
        #     torch_dtype=torch_dtype,
        #     trust_remote_code=trust_remote_code,
        #     model_kwargs=model_kwargs,
        # )
        # self.lm.generation_config.pad_token_id = self.lm.tokenizer.eos_token_id
        # self.profile = FlopsProfiler(self.lm.model)
        self.lm = LLM(model=model)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        # self.profile = FlopsProfiler(self.lm)

        self.history = []

    def generate(self, prompt: str, temperature: float = 0.1, top_p: float = 1, max_tokens: int = 512):
        message =[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': prompt}]
        message = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        output = self.lm.generate(message, sampling_params)
        output = output[0].outputs[0].text
        if self.verbose:
            print(output)
        self.history.append(output)
        return output

    def run(self, prompt: str, time_out: float = 10, temperature: float = 0, top_p: float = 1, max_tokens: int = 512, return_generation=False, repeat=None):
        self.profile.start_profile()
        if repeat is None:
            repeat = self.repeat

        all_output = []
        all_results = {}
        for n in range(repeat):
            output = self.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            all_output.append(output)
            if self.get_answer_symbol is not None:
                match = self.get_answer_symbol.findall(output)
                match = match[0] if match else self.fallback
                if match in all_results:
                    all_results[match] += 1
                else:
                    all_results[match] = 1

        if repeat == 1:
            result = list(all_results.keys())[0]
        else:
            counts = list(all_results.values())
            max_idx = counts.index(max(counts))
            result = list(all_results.keys())[max_idx]
        self.profile.stop_profile()
        flops = self.profile.get_total_flops()
        # macs = self.profile.get_total_macs()
        # params = self.profile.get_total_params()
        # if print_profile:
        #     self.profile.print_model_profile(profile_step=profile_step)
        self.profile.end_profile()
        if return_generation:
            return result, flops, all_output
        return result, flops

if __name__ == "__main__":

    import re
    from tqdm import tqdm
    from datasets import load_dataset

    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    model_str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    system_message = '''
        Write a function to solve a given problem by the user. Only write the program. Do not use `print`.
        The function must be named solution() and return `value` where value is only a number without any signs like '$' or '%'.
        '''
    model = HFProgramInterface(
        system_message=system_message,
        model=model_str,
        get_answer_expr='solution()',
        verbose=True,
        )

    all_scores = []
    for sample in tqdm(gsm8k_test):

        question = sample["question"]
        answer = sample["answer"]
        answer = answer.split("#### ")[-1]
        answer = float(re.findall(r'\d+', answer)[0])
        user_prompt = f"{question}"
        try:
            ans, flops = model.run(user_prompt, temperature=0.1)
            ans = float(ans)
            score = 1 if abs(ans - answer) < 1e-3 else 0
        except Exception as e:
            print("Exception:", e)
            ans = ''
            score = 0
        all_scores.append(score)

    print(f'Accuracy - {sum(all_scores) / len(all_scores)}')
