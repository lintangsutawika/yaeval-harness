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