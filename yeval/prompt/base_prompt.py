from typing import Callable, Optional

class YevalPrompt:
    system_message: str=None
    user_message: Callable=None
    postprocessor: Callable=None

    def __new__(self):
        return self.system_message, self.user_message, self.postprocessor

#     "code": """\
# Solve the following problem by DIRECTLY and ONLY writing a PYTHON program. The answer mush be a function named solution() without any input arguments. The function MUST return an single value.\
# """,
#     "cot_old" : """\
# Solve with the following approach: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.
# """,
#     "cot": """\
# Solve the following problem by thinking step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.\
# """,
#     "routing_nl_first" : """\
# Choose only one way to solve the problem: by thinking step-by-step OR writing a program as a way to solve a given task. Do NOT use both:
# 1. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.
# 2. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.\
# """,
#     "routing_pl_first" : """\
# Choose only one way to solve the problem: by writing a program OR thinking step-by-step as a way to solve a given task. Do NOT use both:
# 1. Writing a program: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, you MUST return an single value.
# 2. Thinking step-by-step: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, you MUST write the answer after 'The answer is'.\
# """,
#     "routing_selection_answer_nl_first" : """\
# Based on a given task, choose only one way that can be used to solve the problem: by natural language OR programming language to solve a given task. Do NOT use both:
# 1. Natural Language: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, if this is the chosen method, answer "natural language".
# 2. Programming Language: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, if this is the chosen method, answer "programming language".\
# """,
#     "routing_selection_answer_pl_first" : """\
# Based on a given task, choose only one way that can be used to solve the problem: by natural language OR programming language to solve a given task. Do NOT use both:
# 1. Programming Language: DIRECTLY and ONLY write a program with the PYTHON programming language. The function must be named solution() without any input arguments. At the end, if this is the chosen method, answer "programming language".\
# 2. Natural Language: Think step-by-step. Derive and go through the logical steps in order to arrive at the final answer. At the end, if this is the chosen method, answer "natural language".
# """,
#     "routing_selection_nl_first" : """\
# Based on a given task, choose only one way that can be used to solve the problem: by natural language OR programming language to solve a given task. Do NOT use both. Answer with either "natural language" or "programming language".
# """,
#     "routing_selection_pl_first" : """\
# Based on a given task, choose only one way that can be used to solve the problem: by programming language OR natural language to solve a given task. Do NOT use both. Answer with either "programming language" or "natural language".
# """,
#           "math_reason" : "Reason step by step and put your final answer within \\boxed{}.",
#     "eng_reason_in_ind" : "Reason step by step in Indonesian and put your final answer within \\boxed{}.",
#     "ind_reason"        : "Berpikir langkah demi langkah dan tuliskan jawaban akhir di dalam \\boxed{}.",
# "eng_reason_analogy_ind": "Use step-by-step reasoning and analogies to explain how to solve the problem using Indonesian. Put your final answer within \\boxed{}.",
#     "eng_reason_in_jpn" : "Reason step by step in Japanese and put your final answer within \\boxed{}.",
#     "jpn_reason"        : "段階的に理論を展開し、最終的な答えを \\boxed{} の中に入れてください。",
# }
