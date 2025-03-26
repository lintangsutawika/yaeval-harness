from yeval.prompt import YevalPrompt, register_prompt

@register_prompt("direct_nl_first")
class DirectNLFirst(YevalPrompt):
    system_message="""\
Choose only one way to solve the problem: by thinking step-by-step OR writing a program as a way to solve a given task. Do NOT use both:
1. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.
2. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.\
"""

@register_prompt("direct_pl_first")
class DirectPLFirst(YevalPrompt):
    system_message="""\
Choose only one way to solve the problem: by writing a program OR thinking step-by-step as a way to solve a given task. Do NOT use both:
1. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
2. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.\
"""

@register_prompt("select_nl_first")
class SelectNLFirst(YevalPrompt):
    system_message="""\
Based on a given task, choose only one way that can be used to solve the problem: by natural language OR programming language to solve a given task. Do NOT use both. Answer with either "natural language" or "programming language".
"""

@register_prompt("select_pl_first")
class SelectPLFirst(YevalPrompt):
    system_message="""\
Based on a given task, choose only one way that can be used to solve the problem: by programming language OR natural language to solve a given task. Do NOT use both. Answer with either "programming language" or "natural language".
"""
