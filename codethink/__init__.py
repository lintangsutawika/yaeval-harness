from .interface import (
    HFProgramInterface,
    HFNatLangInterface,
)

INTERFACE = {
    "code": HFProgramInterface,
    "cot": HFNatLangInterface,
}

SYSTEM_MESSAGE = {
    "code": """\
Solve the problem by DIRECTLY and ONLY writing a program. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "nl" : """\
Solve the problem by thinking step-by-step. Go through the reasoning in order to derive the final answer.
At the end, you MUST write the answer as an integer after '####'."\
"""
}