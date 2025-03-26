
def log_token_usage(state):
    input_tokens = state["usage"]["prompt_tokens"]
    output_tokens = state["usage"]["completion_tokens"]
    total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
