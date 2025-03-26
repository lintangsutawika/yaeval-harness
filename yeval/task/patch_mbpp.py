import whatthepatch

from yeval.task import register_task, YevalTask

def apply_patch(patch_code, base_code):
    patch = list(whatthepatch.parse_patch(patch_code))[0]
    return whatthepatch.apply_diff(patch, base_code)
    updated_code = "\n".join(whatthepatch.apply_diff(patch, base_code))
    return updated_code

def postprocess_patch(x, state):
    print(state["completion"][0])
    original_input = state["full_input"][0]["content"]
    base_code = [line for line in original_input.split("<|diff|>") if line != ""][-1]
    
    if "@@" in base_code:
        current_code = "\n".join(apply_patch(base_code, ""))
    else:
        current_code = ""

    if "@@" in x:
        updated_code = apply_patch(x, current_code)
        code_length = len(updated_code)
        updated_code = [f"@@ -0,0 +1,{code_length} @@"]+[f"+{line}" for line in updated_code]
        updated_code = "\n".join(updated_code)
    else:
        updated_code = base_code

    return updated_code

def preprocess_patch(x, state):
    current_step = state["current_step"]
    base_code = state["step"][current_step-1]["output"][0]
    x = x.split("<|diff|>")[0]+f"<|diff|>{base_code}\n<|diff|>"
    return x, state


# @register_task("patch_mbpp")
class MBPPStep(YevalTask):
    data_path="Muennighoff/mbpp"
    data_name="full"
    input_text=lambda x: f"{x['text']}\n<|diff|>"
    sampling_args={"stop": ["<|diff|>@@"]}
    # sampling_args={"stop": []}
    # exit_loop="<|diff|>\n<|diff|>"
    output_text=lambda x: "0"
    test_split="test"
    evaluation={"pass@1": lambda x,y: -1}
    postprocessor=postprocess_patch

class MBPPNextStep(MBPPStep):
    preprocessor=preprocess_patch

@register_task("patch_mbpp")
class StepByStepMBPPTask(YevalTask):
    subtask_list=[
        MBPPStep,
        MBPPNextStep,
        MBPPNextStep,
        MBPPNextStep,
        MBPPNextStep,
    ]

if __name__ == "__main__":
    pass
