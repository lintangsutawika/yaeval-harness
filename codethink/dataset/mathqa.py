import re
import ast
from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

def mathqa_input(x):
    return f"Question:\n{x['Problem']}\n{x['options']}\nAnswer:"

def match_decimals(prediction, ground_truth):
    reversed_number = str(ground_truth)[::-1]
    decimal_places = reversed_number.find('.')
    decimal_places = decimal_places if decimal_places != -1 else 0
    if decimal_places == 1 and reversed_number[0] == "0":
        decimal_places = 0
    rounded_prediction = round(prediction, decimal_places)
    return rounded_prediction

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

    numeric_answer = option_dict[answer]
    numeric_answer.replace(" . ", "")
    numeric_answer.replace(" : ", "/")
    numeric_answer.replace(" ", "")

    pattern = r"\b(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[/:]\d+)?\b"
    try:
        numeric_answer = re.findall(pattern, numeric_answer)[0]
    except:
        pass

    return " OR ".join([x["correct"], numeric_answer, f"{answer} ) {numeric_answer}"])

def mathqa_fewshot_output(x):
    return f"Let's think step by step. {x['Rationale']} #### {x['correct']}"

def mathqa_eval(prediction, ground_truth):

    score = 0
    letter, number, full_answer = ground_truth.split(" OR ")
    if prediction == letter:
        score = 1
    elif prediction == number:
        score = 1
    elif prediction == full_answer:
        score = 1
    else:
        try:
            if "/" in number:
                gt = eval(number)
            else:
                prediction = float(prediction)
                prediction = match_decimals(prediction, number)
                number = float(number)
                score = 1 if abs(prediction - number) < 1e-3 else 0
        except Exception as e:
            # print(e)
            pass

    return score

MathQADataset = partial(
    TransformedDataset,
    data_path="allenai/math_qa",
    input_text=mathqa_input,
    output_text=partial(mathqa_output),
    fewshot_output_text=mathqa_fewshot_output,
    eval=mathqa_eval,
    test_split="test",
    fewshot_split="train",
)

# MathQALetterAnswerDataset = partial(
#     TransformedDataset,
#     data_path="allenai/math_qa",
#     input_text=mathqa_input,
#     output_text=partial(mathqa_output, letter=True),
#     fewshot_output_text=mathqa_fewshot_output,
#     eval=mathqa_eval,
#     test_split="test",
#     fewshot_split="train",
# )

# MathQANumericAnswerDataset = partial(
#     TransformedDataset,
#     data_path="allenai/math_qa",
#     input_text=mathqa_input,
#     output_text=partial(mathqa_output, letter=False),
#     fewshot_output_text=mathqa_fewshot_output,
#     eval=mathqa_eval,
#     test_split="test",
#     fewshot_split="train",
# )


if __name__ == "__main__":

    dataset = MathQADataset(
        num_fewshot=0,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)