import re

def extract_answer(answer: str):
    try:
        extracted_answer = answer.split('####')[-1].strip()
        if extracted_answer == answer:
            # match = re.search(r"answer is(\w)", answer)
            match = re.search(r"(?i)(?<=answer is ).*", answer)
            if match:
                return match.group(0)
            else:
                return answer
        return extracted_answer
    except Exception as e:
        return answer
