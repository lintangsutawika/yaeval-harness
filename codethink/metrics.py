from transformers import AutoTokenizer


class TokenUsage:
    def __init__(self, model_str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_str)

    def measure(self, generated_strings):
        tokenized_strings = self.tokenizer.encode(generated_strings)
        string_length = len(tokenized_strings)

        return string_length

