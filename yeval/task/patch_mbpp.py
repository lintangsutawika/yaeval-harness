import sys
import subprocess
import whatthepatch

from yeval.task import register_task, YevalTask

def apply_patch(patch_code, base_code):
    try:
        patch = list(whatthepatch.parse_patch(patch_code))[0]
        return whatthepatch.apply_diff(patch, base_code)
    except:
        return ""
    updated_code = "\n".join(whatthepatch.apply_diff(patch, base_code))
    return updated_code

def postprocess_patch(x, state):
    print("output")
    print(x)
    print(state)
    original_input = state["full_input"][0]["content"]
    base_code = [line for line in original_input.split("<|diff|>") if line != ""][-1]

    if "@@" in base_code:
        current_code = "\n".join(apply_patch(base_code, ""))
    else:
        current_code = ""

    if "@@" in x:
        updated_code = "\n".join(apply_patch(x, current_code))
    else:
        updated_code = current_code

    print("Code")
    print(updated_code)
    return updated_code, state

def convert_to_patch(code_snippet):
    if isinstance(code_snippet, str):
        code_snippet = [code_snippet.strip()]
    code_length = len(code_snippet)
    code_patch = "\n".join([f"@@ -0,0 +1,{code_length} @@"]+[f"+{line}" for line in code_snippet])
    return code_patch

def preprocess_patch(x, state):

    current_step = state["current_step"]
    if current_step > 0:
        current_code = state["step"][current_step-1]["output"][0]
        current_code = current_code.split("\n")
        code_length = len(current_code)
        updated_code = "\n".join([f"@@ -0,0 +1,{code_length} @@"]+[f"+{line}" for line in current_code])
        x = x.split("<|diff|>")[0]+f"<|diff|>{updated_code}\n<|diff|>"
    
    print("Query")
    print(x)
    return x, state

def exit_fn(x, state):
    if x.endswith("<|diff|>\n<|diff|>"):
        return True
    else:
        return False

def pass_at_1(completion, test):
    try:
        test_program = completion + "\n" + "\n".join(test)
        subprocess_result = subprocess.run([sys.executable, "-c", test_program], timeout=10, text=True, capture_output=True)
        if subprocess_result.returncode == 0:
            return 1
        return 0
    except Exception as e:
        return 0

@register_task("mbpp_patch_by_patch")
class MBPPStep(YevalTask):
    data_path="evalplus/mbppplus"
    # input_text=lambda x: f"{x['prompt']}\n```\n{convert_to_patch(x["code"].split(":")[0]+":")}\n```\n<|diff|>@@"
    input_text=lambda x: f"{x['prompt']}\n```{x["code"].split(":")[0]+":"}\n```\n<|diff|>@@"
    loop_max=10
    loop_exit=exit_fn
    output_text=lambda x: x["test_list"]
    test_split="test"
    evaluation={"pass@1": pass_at_1}
    preprocessor=preprocess_patch
    postprocessor=postprocess_patch

@register_task("gsm8k_patch_by_patch")
class GSM8kStep(MBPPStep):
    data_path="openai/gsm8k"
    data_name="main"
    # Write a function to solve the following problem.\n
    # input_text=lambda x: f"{x['question']}\n<|diff|>{convert_to_patch('def solution():')}\n<|diff|>"
    # input_text=lambda x: f"{x['question']}\n<|diff|>"
    input_text=lambda x: f"{x['question']}\n```\ndef solution():\n```\n<|diff|>@@"
    # input_text=lambda x: f"{x['question']}"
    # input_text=lambda x: """Write a Python function that takes a string as input, and returns two values: the longest substring without repeating characters, and the total number of unique characters in the string. For example, given the string "abcabcbb", the function should return ("abc", 3). If there are multiple substrings of the same length without repeating characters, any one of them will be sufficient."""
    # sampling_args={"stop": ["<|diff|>", "<|diff|>\n<|diff|>"]}
    # , "include_stop_str_in_output": True
    loop_max=10
    loop_exit=exit_fn
    output_text=lambda x: [f"assert solution() == {x["answer"].split("####")[-1].strip()}"]
    test_split="test"
    # evaluation={"pass@1": lambda x,y: -1}
    evaluation={"pass@1": pass_at_1}
    preprocessor=preprocess_patch
    postprocessor=postprocess_patch


if __name__ == "__main__":
    pass
