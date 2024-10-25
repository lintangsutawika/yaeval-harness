import re
import time
import logging
import transformers

from typing import List

import pal

from vllm import LLM, SamplingParams, RequestOutput

logger = logging.getLogger(__name__)


def get_tokens(model_outputs: RequestOutput):

    all_output_tokens = []
    all_output_text = []
    input_tokens = len(list(model_outputs[0].prompt_token_ids))
    all_output_tokens = 0
    num = len(model_outputs[0].outputs)
    for output in model_outputs[0].outputs:

        output_text = output.text
        output_tokens = len(list(output.token_ids))

        if num == 1:
            return output_text, (input_tokens, output_tokens)

        all_output_text.append(output_text)
        all_output_tokens += output_tokens

    return all_output_text, (input_tokens, all_output_tokens)

class HFProgramInterface(pal.interface.ProgramChatInterface):
    def __init__(self,
                 *args,
                revision="main",
                trust_remote_code=False,
                **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.lm = LLM(
                model=self.model,
                revision=revision,
                trust_remote_code=trust_remote_code,
                )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)

    def generate(self, message, sampling_params):
        output = self.lm.generate(message, sampling_params)
        return output

    def run(self, prompt: str, time_out: float = 10, temperature: float = 0, top_p: float = 1, max_tokens: int = 512, repeat: int = 1, seed: int = None):
        message =[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': prompt}]
        message = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=repeat, seed=seed)
        start_time = time.time()
        output = self.generate(message, sampling_params)
        program, (input_len, output_len) = get_tokens(output)

        all_output = []
        all_results = {}
        all_output_tokens = []
        if isinstance(program, str):
            program = [program]

        for _program in program:
            if self.verbose:
                print(_program)
            self.history.append(_program)
            all_output.append(_program)
            all_output_tokens.append(_program)
            code = self.process_generation_to_code(_program)

            with pal.interface.timeout(time_out):
                try:
                    exec_result = self.execute(code)
                except Exception as e:
                    print(e)
                    exec_result = ""
        
            if exec_result in all_results:
                all_results[exec_result] += 1
            else:
                all_results[exec_result] = 1

        if repeat == 1:
            result = list(all_results.keys())[0]
        else:
            counts = list(all_results.values())
            max_idx = counts.index(max(counts))
            result = list(all_results.keys())[max_idx]

        duration = time.time() - start_time
        output_dict = {
            "input_len": input_len,
            "output_len": output_len,
            "duration": duration,
            "system_output": output,
        }
        return result, output_dict


class HFNatLangInterface:
    def __init__(self,
                 model,
                 system_message,
                 repeat=1,
                 get_answer_symbol=None,
                 fallback="[INVALID]",
                 verbose=False,
                 revision="main",
                 trust_remote_code=False,
                 **kwargs):

        self.system_message = system_message
        self.repeat = repeat
        if isinstance(get_answer_symbol, str):
            self.get_answer_symbol = [re.compile(get_answer_symbol)]
        else:
            self.get_answer_symbol = [re.compile(pattern) for pattern in get_answer_symbol]
        self.fallback = fallback
        self.verbose = verbose

        self.lm = LLM(
            model=model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        self.history = []

    def generate(self, message, sampling_params):
        output = self.lm.generate(message, sampling_params)
        return output

    def run(self, prompt: str, time_out: float = 10, temperature: float = 0, top_p: float = 1, max_tokens: int = 512, repeat: int = 1, seed: int = None):
        message =[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': prompt}]
        message = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=repeat, seed=seed)
        start_time = time.time()
        output = self.generate(message, sampling_params)
        output, (input_len, output_len) = get_tokens(output)

        all_output = []
        all_results = {}
        all_output_tokens = []
        if isinstance(output, str):
            output = [output]

        for _output in output:
            if self.verbose:
                print(_output)
            self.history.append(_output)
            all_output.append(_output)
            all_output_tokens.append(output_len)
            if self.get_answer_symbol is not None:
                match = self.fallback
                for get_answer_symbol in self.get_answer_symbol:
                    _match = get_answer_symbol.findall(_output)
                    if _match:
                        match = _match[0]
                        break
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

        duration = time.time() - start_time
        output_dict = {
            "input_len": input_len,
            "output_len": output_len,
            "duration": duration,
            "system_output": output,
        }
        return result, output_dict

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
