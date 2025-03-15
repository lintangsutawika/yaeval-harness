import re
import ast
import math
from functools import partial

#from sympy import *
#from latex2sympy2 import latex2sympy

from codethink.dataset import register_task

from codethink._task import Task, create_task
from codethink._data import TransformedDataset

from codethink.utils import is_runnable_code

from datasets import concatenate_datasets

def math_merge(dataset):
    return concatenate_datasets(concatenate_datasets)


def math_level_5(dataset):
    return dataset.filter(lambda x: x["level"] == "Level 5")

def math_input(x):
    return x['problem']

def math_output(x):
    return remove_boxed(last_boxed_only_string(x["solution"]))

def math_eval(prediction, ground_truth):

    score = 0
    try:
        for sym in ["\\$", ",\\!"]:
            ground_truth = ground_truth.replace(sym, "")
        #if math.isclose(eval(prediction), eval(ground_truth)):
        #    score = 1
        if math.isclose(
            N(latex2sympy(prediction)),
            N(latex2sympy(ground_truth))
            ):
            score = 1
    except:
        pass

    if is_equiv(prediction, ground_truth):
        score = 1

    return score

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

#def remove_boxed(s):
#    if s.endswith("$"):
#        s = s[:-1]
#    if s.startswith("$"):
#        s = s[1:]
#    match = re.search(r"\\boxed{(.*?)}", s)
#    return match.group(1) if match else s

def remove_boxed(s):

    if s.endswith("$"):
        s = s[:-1]
    if s.startswith("$"):
        s = s[1:]

    s = s.split("\\(")[-1]
    s = s.split("\\)")[0]

    if "\\boxed" not in s:
        return s

    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    try:
        s = re.search(r"\\boxed{(.*?)}", s)[0]
    except:
        pass

    left = "\\boxed{"
    # s = s.split(left)[1:]
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]
    except:
        return s

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

mathdatasets = {}
for data_name in ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']:
    mathdatasets[f"math-{data_name}"] = partial(
        TransformedDataset,
        data_path="EleutherAI/hendrycks_math",
        data_name=data_name,
        preprocessing=math_level_5,
        input_text=math_input,
        output_text=partial(math_output),
        test_split="test",
        fewshot_split="train",
    )

def preprocess_routing(x, state):
    current_step = state["current_step"]
    solve_with = state["step"][current_step-1]["output"].lower()
    if "programming language" in solve_with:
        state["system_message"] = "code"
    else:
        state["system_message"] = "cot"
    return x, state

math_operands = [
    ["**", "^"],
    ["*", "×"],
    ]

def postprocess_routing(x, state):
    x = x["response"][0]
    exec_result = is_runnable_code(x)
    if exec_result:
        return exec_result, state
    else:
        try:
            for answer_start in ["answer is", "is:", "swer:", "is", "swer"]:
                if answer_start in x:
                    x = x.split(answer_start)[-1].strip()
                    break
            if x.endswith("."):
                x = x[:-1]
        except:
            pass
    x = remove_boxed(x)
    for sym, opr in math_operands:
        x = x.replace(sym, opr)
    return x, state

@register_task(
    "math_algebra",
    dataset=mathdatasets["math-algebra"],
    postprocessor=postprocess_routing,
    evaluation={"accuracy": math_eval},
    )
class MathAlgebra(Task):
    pass

@register_task(
    "math_counting_and_probability",
    dataset=mathdatasets["math-counting_and_probability"],
    postprocessor=postprocess_routing,
    evaluation={"accuracy": math_eval},
    )
class MathCountingAndProbability(Task):
    pass

@register_task(
    "math_geometry",
    dataset=mathdatasets["math-geometry"],
    postprocessor=postprocess_routing,
    evaluation={"accuracy": math_eval},
    )
class MathGeometry(Task):
    pass

@register_task(
    "math_intermediate_algebra",
    dataset=mathdatasets["math-intermediate_algebra"],
    postprocessor=postprocess_routing,
    evaluation={"accuracy": math_eval},
    )
class MathIntermediateAlgebra(Task):
    pass

@register_task(
    "math_number_theory",
    dataset=mathdatasets["math-number_theory"],
    postprocessor=postprocess_routing,
    evaluation={"accuracy": math_eval},
    )
class MathNumberTheory(Task):
    pass

@register_task(
    "math_prealgebra",
    dataset=mathdatasets["math-prealgebra"],
    postprocessor=postprocess_routing,
    evaluation={"accuracy": math_eval},
    )
class MathPrealgebra(Task):
    pass

@register_task(
    "math_precalculus",
    dataset=mathdatasets["math-precalculus"],
    postprocessor=postprocess_routing,
    evaluation={"accuracy": math_eval},
    )
class MathPrecalculus(Task):
    pass

@register_task(
    "math_algebra-routing_pipeline_pl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-algebra"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-algebra",
            dataset=mathdatasets["math-algebra"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathAlgebraRouting(Task):
    pass

@register_task(
    "math_counting_and_probability-routing_pipeline_pl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-counting_and_probability"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-counting_and_probability",
            dataset=mathdatasets["math-counting_and_probability"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathCountingAndProbabilityRouting(Task):
    pass

@register_task(
    "math_geometry-routing_pipeline_pl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-geometry"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-geometry",
            dataset=mathdatasets["math-geometry"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathGeometryRouting(Task):
    pass

@register_task(
    "math_intermediate_algebra-routing_pipeline_pl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-intermediate_algebra"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-intermediate_algebra",
            dataset=mathdatasets["math-intermediate_algebra"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathIntermediateAlgebraRouting(Task):
    pass

@register_task(
    "math_number_theory-routing_pipeline_pl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-number_theory"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-number_theory",
            dataset=mathdatasets["math-number_theory"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathNumberTheoryRouting(Task):
    pass

@register_task(
    "math_prealgebra-routing_pipeline_pl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-prealgebra"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-prealgebra",
            dataset=mathdatasets["math-prealgebra"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathPrealgebraRouting(Task):
    pass

@register_task(
    "math_precalculus-routing_pipeline_pl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-precalculus"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_pl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-precalculus",
            dataset=mathdatasets["math-precalculus"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathPrecalculusRouting(Task):
    pass

@register_task(
    "math_algebra-routing_pipeline_nl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-algebra"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_nl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-algebra",
            dataset=mathdatasets["math-algebra"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathAlgebraRoutingNL(Task):
    pass

@register_task(
    "math_counting_and_probability-routing_pipeline_nl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-counting_and_probability"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_nl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-counting_and_probability",
            dataset=mathdatasets["math-counting_and_probability"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathCountingAndProbabilityRoutingNL(Task):
    pass

@register_task(
    "math_geometry-routing_pipeline_nl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-geometry"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_nl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-geometry",
            dataset=mathdatasets["math-geometry"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathGeometryRoutingNL(Task):
    pass

@register_task(
    "math_intermediate_algebra-routing_pipeline_nl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-intermediate_algebra"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_nl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-intermediate_algebra",
            dataset=mathdatasets["math-intermediate_algebra"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathIntermediateAlgebraRoutingNL(Task):
    pass

@register_task(
    "math_number_theory-routing_pipeline_nl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-number_theory"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_nl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-number_theory",
            dataset=mathdatasets["math-number_theory"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathNumberTheoryRoutingNL(Task):
    pass

@register_task(
    "math_prealgebra-routing_pipeline_nl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-prealgebra"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_nl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-prealgebra",
            dataset=mathdatasets["math-prealgebra"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathPrealgebraRoutingNL(Task):
    pass

@register_task(
    "math_precalculus-routing_pipeline_nl_first",
    subtask_list=[
        create_task(
            name="routing",
            dataset=partial(
                mathdatasets["math-precalculus"],
                input_text=lambda x: math_input(x)+"\n\nWhich method is the best way to solve this problem?",
            ),
            system_message="routing_selection_nl_first",
            sampling_args={"stop": ["\n\n", "\n"]},
        ),
        create_task(
            name="math-precalculus",
            dataset=mathdatasets["math-precalculus"],
            preprocessor=preprocess_routing,
            postprocessor=postprocess_routing,
            evaluation={"accuracy": math_eval},
        ),
    ])
class MathPrecalculusRoutingNL(Task):
    pass


from codethink.dataset.utils import remove_boxed, last_boxed_only_string

@register_task(
    "math_algebra_train",
    dataset=partial(
        mathdatasets["math-algebra"],
        preprocessing=None,
        test_split="train",
        ),
    postprocessor=lambda x, state: (remove_boxed(last_boxed_only_string(x['response'][0])), state),
    evaluation={"accuracy": math_eval},
    )
class MathAlgebraTrain(Task):
    pass

if __name__ == "__main__":

    dataset = mathdatasets["math-algebra"]()

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)
