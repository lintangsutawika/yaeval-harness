import numpy as np

def log_token_usage(state):
    try:
        input_tokens = state["usage"]["prompt_tokens"]
        output_tokens = state["usage"]["completion_tokens"]
        total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    except KeyError:
        return {}

def log_logprob(state):
    try:
        choice_logprob = []
        for choice in state['choices']:
            if "content" in choice['logprobs']:
                logprob_list = [token['logprob'] for token in choice['logprobs']['content']]
            elif "token_logprobs" in choice['logprobs']:
                logprob_list = choice["logprobs"]["token_logprobs"]
            choice_logprob.append(sum(logprob_list)/len(logprob_list))
            # choice_logprob.append(sum(logprob_list))
        return {"logprob": choice_logprob}
    except KeyError:
        return {}