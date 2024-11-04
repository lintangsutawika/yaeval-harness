import re
import ast
from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

def mathqa_input(x):
    return f"Question:\n{x["Problem"]}\n{x["options"]}\nAnswer:"

def mathqa_output(x):

    answer = x["correct"]
    if x['options'].startswith("["):
        option_list = ast.literal_eval(x['options'])
        option_dict = {option.split(" ) ")[0]: option.split(" ) ")[1] for option in [option for option in option_list]}
    else:
        option_dict = {}
        letter_choice = ["a", "b", "c", "d", "e"]
        for idx, letter in enumerate(letter_choice):
            choice = x['options'].split(f"{letter} ) ")[-1]
            if idx < len(letter_choice)-1:
                choice = choice.split(f" , {letter_choice[idx+1]} ) ")[0]
            option_dict[letter] = choice

    return answer, option_dict[answer]

def mathqa_fewshot_output(x):
    return f"Let's think step by step. {x["Rationale"]} #### {x["correct"]}"

def mathqa_eval(prediction, ground_truth):
    try:
        score = 0
        letter, number = ground_truth
        if prediction == letter:
            score = 1
        elif prediction == number:
            score = 1
        else:
            prediction = ast.literal_eval(prediction)
            number.replace(":". "/")

            try:
                ground_truth = ast.literal_eval(number)
            except:
                try:
                    ground_truth = ast.literal_eval(eval(number))
                except:
                    ground_truth = "".join(re.findall(r'\d+', number))
                    ground_truth = ast.literal_eval(ground_truth)
            
            prediction = type(ground_truth)(prediction)
            score = 1 if abs(prediction - ground_truth) < 1e-2 else 0

    except Exception as e:
        print("Exception:", e)
        score = 0

    return score

MathQADataset = partial(
    TransformedDataset,
    data_path="allenai/math_qa",
    input_text=mathqa_input,
    output_text=mathqa_output,
    # fewshot_output_text=mathqa_fewshot_output,
    eval=mathqa_eval,
    test_split="test",
    fewshot_split="train",
)

if __name__ == "__main__":

    dataset = MathQADataset(
        num_fewshot=5,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)