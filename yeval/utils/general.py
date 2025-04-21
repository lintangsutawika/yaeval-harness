import re
import sys
import requests
import subprocess
import pandas as pd
# from zeno_client import ZenoClient, ZenoMetric
# TODO: Function to fill in results to a csv or google spreadsheet
import os
import glob
import importlib
import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)

from functools import partial


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
        k: handle_arg_string(v) if k != "extra_body" else eval(v) for k, v in [arg.split("=") for arg in arg_list]
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

def check_api_health(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

def extract_fn(answer: str):
    try:
        extracted_answer = answer.split('####')[-1].strip()
        if extracted_answer == answer:
            match = re.search(r"answer is(\w)", answer)
            if match:
                return match.group(1)
            else:
                return answer
        return extracted_answer
    except:
	    return answer

def import_modules(path=None):

    if path is None:
        path = os.path.dirname(__file__)

    module_files = glob.glob(
        os.path.join(
            path, "**", "*.py"
            ), recursive=True
        )

    for file in module_files:
        module_name = os.path.basename(file)[:-3]
        if module_name not in ["__init__", "__main__"] and module_name.isidentifier():
            try:
                spec = importlib.util.spec_from_file_location(f"{module_name}", file)
                foo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(foo)
                # importlib.import_module(f".{module_name}", package=__name__)
            except Exception as e:
                logging.warning(f"{file}: {e}")

def parse_tree(task_str):
    def split_top_level(s, sep):
        parts = []
        current = ''
        depth = 0
        for char in s:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == sep and depth == 0:
                parts.append(current.strip())
                current = ''
            else:
                current += char
        if current:
            parts.append(current.strip())
        return parts

    def parse_subtree(s):
        if 't//' in s:
            root, rest = s.split('t//', 1)
            children = split_top_level(rest, '+')
            parsed_children = []
            for child in children:
                if child.startswith('(') and child.endswith(')'):
                    parsed_children.append(parse_subtree(child[1:-1]))
                else:
                    parsed_children.append(parse_subtree(child))
            return [root] + parsed_children
        else:
            return [s]

    return parse_subtree(task_str)

def list_to_task_dict(tree):
    node = {"task": tree[0]}
    if len(tree) > 1:
        node["subtask"] = [list_to_task_dict(child) for child in tree[1:]]
    return node


# In [72]: def instantiate_task_tree(tree, TASK_LIST):
#     ...:     task_name = tree[0]
#     ...:     subtask_instances = []
#     ...:
#     ...:     if len(tree) > 1:
#     ...:         for child in tree[1:]:
#     ...:             print(child)
#     ...:             subtask_instance = instantiate_task_tree(child, TASK_LIST)
#     ...:             subtask_instances.append(subtask_instance)
#     ...:             print(subtask_instance)
#     ...:
#     ...:     return TASK_LIST[task_name](subtask_list=subtask_instances)
