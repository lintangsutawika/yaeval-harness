import re
from yeval.response.code_responses import is_runnable_code

def match_routing(prediction, ground_truth):
    prediction = prediction.strip().lower()
    ground_truth = ground_truth.strip().lower()
    if re.sub(r'[^\w\s]', '', prediction) == re.sub(r'[^\w\s]', '', ground_truth):
        return 1
    elif ground_truth in prediction:
        return 1
    return 0

def preprocess_routing(x, state, pl_system_message, nl_system_message):
    current_step = state["current_step"]
    solve_with = state["step"][current_step-1]["output"][0].split("\n")[0]
    if solve_with == "programming language":
        state["system_message"] = pl_system_message
    elif solve_with == "natural language":
        state["system_message"] = nl_system_message
    return x, state

def postprocess_routing(x, state):
    current_step = state["current_step"]
    solve_with = state["step"][current_step-1]["output"][0]
    if solve_with == "programming language":
        x = is_runnable_code(x) 
    elif solve_with == "natural language":
        try:
            x = x.split("answer is")[-1].strip()
        except:
            pass
    return x, state
