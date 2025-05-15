import numpy as np

def log_token_usage(state):
    input_tokens = state["usage"]["prompt_tokens"]
    output_tokens = state["usage"]["completion_tokens"]
    total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

def log_logprob(state):
    for choice in state['choices']:
        choice_logprob = sum([token['logprob'] for token in choice['logprobs']['content']])
        return {"logprob": choice_logprob}