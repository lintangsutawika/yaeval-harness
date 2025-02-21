import re

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
