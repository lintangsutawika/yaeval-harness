import os

def execute(x):
    pass

def extract_regex(answer: str, fallback: str, regex: List):
    match = fallback
    for _regex in regex:
        _match = _regex.findall(answer)
        if _match:
            match = _match[0]
            break
    return match

def extract_fn(answer: str, fallback: str):
    answer = answer.split('####')[-1].strip()
    for char in [',', '$', '%', 'g']:
        answer = answer.replace(char, '')
    try:
        return answer
    except:
        return fallback

def pass_fn(answer: str, fallback: str):
    try:
        return answer.strip()
    except:
        return fallback

# self.get_answer_symbol = partial(extract_regex, fallback=fallback, regex=[re.compile(pattern) for pattern in get_answer_symbol])
