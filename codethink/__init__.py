# from vllm import ModelRegistry
# from .modeling import AutoModelforCodeSolverLM
# ModelRegistry.register_model("AutoModelforCodeSolverLM", AutoModelforCodeSolverLM)

from .interface import (
    # HFProgramInterface,
    SolverInterface,
)

INTERFACE = {
    # "code": HFProgramInterface,
    "default": SolverInterface,
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
Choose only one way to solve the problem: by thinking step-by-step or writing a program as a way to solve a given task. Do NOT use both:
1. Thinkng step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after '####'.
2. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.\
""",
    "routing-code-first" : """\
Choose only one way to solve the problem: by writing a program or thinking step-by-step as a way to solve a given task. Do NOT use both:
1. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
2. Thinkng step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after '####'.\
""",
}