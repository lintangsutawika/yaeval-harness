# from vllm import ModelRegistry
# from .modeling import AutoModelforCodeSolverLM
# ModelRegistry.register_model("AutoModelforCodeSolverLM", AutoModelforCodeSolverLM)

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
    "code-comment": """\
Solve the problem by DIRECTLY and ONLY writing a program. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "code-python": """\
Solve the problem by DIRECTLY and ONLY writing a program with the PYTHON programming language. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code-comment-python": """\
Solve the problem by DIRECTLY and ONLY writing a program with the PYTHON programming language. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "code-javascript": """\
Solve the problem by DIRECTLY and ONLY writing a program with the JAVASCRIPT programming language. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code-comment-javascript": """\
Solve the problem by DIRECTLY and ONLY writing a program with the JAVASCRIPT programming language. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "code-rust": """\
Solve the problem by DIRECTLY and ONLY writing a program with the RUST programming language. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code-comment-rust": """\
Solve the problem by DIRECTLY and ONLY writing a program with the RUST programming language. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "code-c": """\
Solve the problem by DIRECTLY and ONLY writing a program with the C programming language. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code-comment-c": """\
Solve the problem by DIRECTLY and ONLY writing a program with the C programming language. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "code-cpp": """\
Solve the problem by DIRECTLY and ONLY writing a program with the C++ programming language. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code-comment-cpp": """\
Solve the problem by DIRECTLY and ONLY writing a program with the C++ programming language. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "cot" : """\
Solve the problem by thinking step-by-step. Go through the reasoning in order to derive the final answer.
At the end, you MUST write the answer as an integer after '####'."\
""",
    "routing-cot-first" : """\
Choose to solve the problem by either:
1. Thinking step-by-step. Go through the reasoning in order to derive the final answer. At the end, you MUST write the answer as an integer after '####'.
2. DIRECTLY and ONLY writing a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
""",
    "routing-code-first" : """\
Choose to solve the problem by either:
1. DIRECTLY and ONLY writing a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
2. Thinking step-by-step. Go through the reasoning in order to derive the final answer. At the end, you MUST write the answer as an integer after '####'.
""",
}