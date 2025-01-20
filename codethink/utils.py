import sys
import subprocess
import pandas as pd
# from zeno_client import ZenoClient, ZenoMetric
# TODO: Function to fill in results to a csv or google spreadsheet


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }
    return args_dict


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def calculate_tokens(tokens):
    if isinstance(tokens, Tuple):
        tokens = [tokens]

    num_tokens = 0
    for _tokens in tokens:
        i_tokens, o_tokens = _tokens
        num_tokens = len(i_tokens + o_tokens)

    return num_tokens


def zeno_upload(run_name, result_dict, metric_column="score"):


    client = ZenoClient("zen_ItYuaijqhVoxmHR_ScDYpM43OBvq0eO1dw7FrE8o9gI")
    df = pd.DataFrame(data=result_dict)

    # Create a project.
    project = client.create_project(
        name=run_name,
        view={
            "data": {
                "type": "text"
            },
            "label": {
                "type": "text"
            },
            "output": {
                "type": "code"
            }
        },
        metrics=[
            ZenoMetric(name="avg_score", type="mean", columns=["score"]),
            ZenoMetric(name="avg_duration", type="mean", columns=["duration"]),
            ZenoMetric(name="avg_input_tokens", type="mean", columns=["input_tokens"]),
            ZenoMetric(name="avg_output_tokens", type="mean", columns=["output_tokens"]),
            ZenoMetric(name="avg_total_tokens", type="mean", columns=["total_tokens"]),
        ]
    )

    # Upload the data.
    project.upload_dataset(df, id_column="idx", data_column="user_input", label_column="answer")
    project.upload_system(df[["idx", "system_output"]], name="System A", id_column="idx", output_column="system_output")

def process_generation_to_code(gens: str, answer_expr: str):
    if '```python' in gens:
        gens = gens.split('```python')[1].split('```')[0]
    elif '```' in gens:
        gens = gens.split('```')[1].split('```')[0]
    elif answer_expr in gens:
        gens = "def "+answer_expr+f"{answer_expr}".join(gens.split(answer_expr)[1:])
    else:
        return False
        
    return gens.split('\n')

def is_runnable_code(text_string, answer_expr='solution()', time_out=10):
    # Check if the _output is a program
    code = process_generation_to_code(text_string, answer_expr)
    if code:
        def _generate_code(code, answer_expr):
            return "\n".join(code)+f"\nans = 'ans='+str({answer_expr})\nprint(ans)"
        # Generate code snippet that will be executed in a different process
        code_snippet = _generate_code(code, answer_expr)
        try:
            subprocess_result = subprocess.run([sys.executable, "-c", code_snippet], timeout=time_out, text=True, capture_output=True)
            exec_result = subprocess_result.stdout.split("ans=")[-1].strip()
            return exec_result
        except Exception as e:
            return False
    else:
        return False

