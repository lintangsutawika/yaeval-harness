from yeval.prompt import YevalPrompt, register_prompt

@register_prompt("code")
class CodePrompt(YevalPrompt):
    system_message="""\
Solve the following problem by DIRECTLY and ONLY writing a PYTHON program. \
The answer mush be a function named solution() without any input arguments. \
The function MUST return an single value.\
"""
    postprocessor="code"

@register_prompt("cot")
class CotPrompt(YevalPrompt):
    system_message="""\
Solve the following problem by thinking step-by-step. \
Derive and go through the logical steps in order to arrive at the final answer. \
At the end, you MUST write the answer after 'The answer is'.\
"""
    postprocessor="cot"