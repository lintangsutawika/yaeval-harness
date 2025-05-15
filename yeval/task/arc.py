from yeval.task import register_task, YevalTask

def mcq_input(x):
    choices = x["choices"]
    choice_list = [f"{a}. {b}" for a,b in list(zip(choices["label"], choices["text"]))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either A, B, C, or D.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def mcq_output(x):
    label = x['answerKey']
    label_idx = x['choices']['label'].index(label)
    text = x['choices']['text'][label_idx]
    return [label, text, f"{label}. {text}"]

def mcq_eval(prediction, ground_truth):
    score = 0
    try:
        letter, text, full_span = ground_truth
        if full_span == ground_truth:
            return 1
        prediction = prediction.split(".")[0]
        if prediction in ["A", "B", "C", "D"]:
            if prediction == letter:
                score = 1
        elif prediction == text:
            score = 1
    except Exception as e:
        pass
    return score

@register_task("arc_challenge")
class ARCChallengeTask(YevalTask):
    data_path="allenai/ai2_arc"
    data_name="ARC-Challenge"
    input_text=mcq_input
    output_text=mcq_output
    test_split="test"
    evaluation={"accuracy": mcq_eval}

if __name__ == "__main__":
    pass
