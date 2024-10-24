import re
import time
import logging
import transformers

import pal

from vllm import LLM, SamplingParams, RequestOutput

logger = logging.getLogger(__name__)


def get_tokens(model_outputs: RequestOutput):
    input_tokens = list(model_outputs[0].prompt_token_ids)
    output_tokens = list(model_outputs[0].outputs[0].token_ids)
    output_text = model_outputs[0].outputs[0].text

    return output_text, (input_tokens, output_tokens)

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

        self.lm = LLM(
                model=self.model,
                revision=revision,
                trust_remote_code=trust_remote_code,
                )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)

    def generate(self, prompt: str, temperature: float = 0.1, top_p: float = 1, max_tokens: int = 512):
        message =[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': prompt}]
        message = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        output = self.lm.generate(message, sampling_params)
        return output

    def run(self, prompt: str, time_out: float = 10, temperature: float = 0, top_p: float = 1, max_tokens: int = 512):
        start_time = time.time()
        output = self.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        program, tokens = get_tokens(output)
        input_len, output_len = tokens
        if self.verbose:
            print(program)
        self.history.append(program)
        code = self.process_generation_to_code(program)

        with pal.interface.timeout(time_out):
            try:
                exec_result = self.execute(code)
            except Exception as e:
                print(e)
                return "", computed_tokens, code
        
        duration = time.time() - start_time
        output_dict = {
            "input_len": input_len,
            "output_len": output_len,
            "duration": duration,
            "generation": code,
        }
        return exec_result, output_dict


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
        self.get_answer_symbol = re.compile(get_answer_symbol)
        self.fallback = fallback
        self.verbose = verbose

        self.lm = LLM(
            model=model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
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
        return output

    def run(self, prompt: str, time_out: float = 10, temperature: float = 0, top_p: float = 1, max_tokens: int = 512, repeat=None):
        start_time = time.time()
        if repeat is None:
            repeat = self.repeat

        all_output = []
        all_results = {}
        all_tokens = []
        for n in range(repeat):
            output = self.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            output, tokens = get_tokens(output)
            input_len, output_len = tokens
            if self.verbose:
                print(output)
            self.history.append(output)
            all_output.append(output)
            all_tokens.append(tokens)
            if self.get_answer_symbol is not None:
                match = self.get_answer_symbol.findall(output)
                match = match[0] if match else self.fallback
                if match in all_results:
                    all_results[match] += 1
                else:
                    all_results[match] = 1

        if repeat == 1:
            result = list(all_results.keys())[0]
            computed_tokens = all_tokens[0]
        else:
            counts = list(all_results.values())
            max_idx = counts.index(max(counts))
            result = list(all_results.keys())[max_idx]
            computed_tokens = all_tokens

        duration = time.time() - start_time
        output_dict = {
            "input_len": input_len,
            "output_len": output_len,
            "duration": duration,
            "generation": output,
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
