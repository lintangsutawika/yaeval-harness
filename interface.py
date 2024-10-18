import torch
import transformers

import pal


class LlamaProgramInterface(pal.interface.ProgramChatInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = transformers.pipeline(
            "text-generation",
            model=self.model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",

        )

    def generate(self, prompt: str, temperature: float = 0.1, top_p: float = 1, max_tokens: int = 512):
        messages =[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': prompt}]
        outputs = self.model(messages, temperature=temperature, top_p=top_p, max_new_tokens=max_tokens)
        program = outputs[0]["generated_text"][-1]['content']
        if self.verbose:
            print(program)
        self.history.append(program)
        return self.process_generation_to_code(program)


if __name__ == "__main__":

    import re
    from tqdm import tqdm
    from datasets import load_dataset

    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    model_str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = LlamaProgramInterface(model=model_str)

    all_score = []
    i = 0
    for sample in tqdm(gsm8k_test):

        question = sample["question"]
        answer = sample["answer"]
        answer = answer.split("#### ")[-1]
        answer = re.findall(r'\d+', answer)[0]
        user_prompt = f"Write a function to solve the following problem, only write the program. The function must return `### value`:\n{question}"

        output = model.generate(user_prompt)
        print(output)
        break