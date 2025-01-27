SYSTEM_MESSAGE = {
    "none": "",
    "code": """\
Solve the problem by DIRECTLY and ONLY writing a program. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code_comment": """\
Solve the problem by DIRECTLY and ONLY writing a program. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "code_python": """\
Solve the problem by DIRECTLY and ONLY writing a program with the PYTHON programming language. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code_comment_python": """\
Solve the problem by DIRECTLY and ONLY writing a program with the PYTHON programming language. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "code_javascript": """\
Solve the problem by DIRECTLY and ONLY writing a program with the JAVASCRIPT programming language. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code_comment_javascript": """\
Solve the problem by DIRECTLY and ONLY writing a program with the JAVASCRIPT programming language. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "code_rust": """\
Solve the problem by DIRECTLY and ONLY writing a program with the RUST programming language. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code_comment_rust": """\
Solve the problem by DIRECTLY and ONLY writing a program with the RUST programming language. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "code_c": """\
Solve the problem by DIRECTLY and ONLY writing a program with the C programming language. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code_comment_c": """\
Solve the problem by DIRECTLY and ONLY writing a program with the C programming language. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "code_cpp": """\
Solve the problem by DIRECTLY and ONLY writing a program with the C++ programming language. The function must be named solution() without any input arguments.
At the end, you MUST return an single value.\
""",
    "code_comment_cpp": """\
Solve the problem by DIRECTLY and ONLY writing a program with the C++ programming language. The function must be named solution() without any input arguments.
Explain your reasoning by adding comments in the program.
At the end, you MUST return an single value.\
""",
    "cot" : """\
Solve the problem by thinking step-by-step. Go through the reasoning in order to derive the final answer.
At the end, you MUST write the answer after 'The answer is'."\
""",
    "routing_nl_first" : """\
Choose only one way to solve the problem: by thinking step-by-step OR writing a program as a way to solve a given task. Do NOT use both:
1. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.
2. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.\
""",
    "routing_pl_first" : """\
Choose only one way to solve the problem: by writing a program OR thinking step-by-step as a way to solve a given task. Do NOT use both:
1. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
2. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.\
""",
    "routing_selection_answer_nl_first" : """\
Based on a given task, choose only one way that can be used to solve the problem: by natural language OR programming language to solve a given task. Do NOT use both:
1. Natural Language: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, if this is the chosen method, answer "natural language".
2. Programming Language: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, if this is the chosen method, answer "programming language".\
""",
    "routing_selection_answer_pl_first" : """\
Based on a given task, choose only one way that can be used to solve the problem: by natural language OR programming language to solve a given task. Do NOT use both:
1. Programming Language: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, if this is the chosen method, answer "programming language".\
2. Natural Language: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, if this is the chosen method, answer "natural language".
""",
    "routing_selection_nl_first" : """\
Based on a given task, choose only one way that can be used to solve the problem: by natural language OR programming language to solve a given task. Do NOT use both. Answer with either "natural language" or "programming language".
""",
    "routing_selection_pl_first" : """\
Based on a given task, choose only one way that can be used to solve the problem: by programming language OR natural language to solve a given task. Do NOT use both. Answer with either "programming language" or "natural language".
""",
}
