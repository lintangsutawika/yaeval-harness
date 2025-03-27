import sys
import subprocess
import whatthepatch

from yeval.task import register_task, YevalTask

def apply_patch(patch_code, base_code):
    patch = list(whatthepatch.parse_patch(patch_code))[0]
    return whatthepatch.apply_diff(patch, base_code)
    updated_code = "\n".join(whatthepatch.apply_diff(patch, base_code))
    return updated_code

def postprocess_patch(x, state):
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

    # print("Code")
    # print(updated_code)
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
    
    # print("Query")
    # print(x)
    return x, state

def exit_fn(x, state):
    if x.endswith("<|diff|>\n<|diff|>"):
        return True
    else:
        return False

def pass_at_1(completion, test):
    # print(completion)
    try:
        test_program = completion + "\n" + "\n".join(test)
        subprocess_result = subprocess.run([sys.executable, "-c", test_program], timeout=10, text=True, capture_output=True)
        # print(subprocess_result)
        if subprocess_result.returncode == 0:
            return 1
        return 0
    except Exception as e:
        return 0

@register_task("mbpp_patch_by_patch")
class MBPPStep(YevalTask):
    data_path="evalplus/mbppplus"
    input_text=lambda x: f"{x['prompt']}\n<|diff|>{convert_to_patch(x["code"].split(":")[0]+":")}\n<|diff|>"
    sampling_args={"stop": ["<|diff|>@@"]}
    loop_max=10
    loop_exit=exit_fn
    output_text=lambda x: x["test_list"]
    test_split="test"
    # evaluation={"pass@1": lambda x,y: -1}
    evaluation={"pass@1": pass_at_1}
    preprocessor=preprocess_patch
    postprocessor=postprocess_patch

# class MBPPNextStep(MBPPStep):
    

# @register_task("patch_mbpp")
# class StepByStepMBPPTask(YevalTask):
#     subtask_list=[
#         MBPPStep,
#         MBPPNextStep,
#         MBPPNextStep,
#         MBPPNextStep,
#         MBPPNextStep,
#     ]

if __name__ == "__main__":
    pass
