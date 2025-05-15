import ast

"""
Example response from OpenAI Chat Completion
{
  "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
  "object": "chat.completion",
  "created": 1741569952,
  "model": "gpt-4o-2024-08-06",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?",
        "refusal": null,
        "annotations": []
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 10,
    "total_tokens": 29,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "audio_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
  "service_tier": "default"
}
"""
def vllm_postprocess(response):
    """
    Returns list of responses and the original response object
    """
    response_dict = ast.literal_eval(
        response.model_dump_json(
            exclude_unset=True,
            exclude_none=True
            )
        )
    return (
        [
            response.choices[i].message.content
            for i in range(0, len(response.choices))
        ],
        response_dict
    )

def openai_completion_postprocess(response):
    """
    Returns list of responses and the original response object
    """
    response_dict = ast.literal_eval(
        response.model_dump_json(
            exclude_unset=True,
            exclude_none=True
            )
        )
    return (
        [
            response.choices[i].text
            for i in range(0, len(response.choices))
        ],
        response_dict
    )
