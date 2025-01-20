import re
from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

def gsm8k_fewshot_input(x):
    fewshot_context = """\
Question:\nThere are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:\nThere are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6.

Question:\nIf there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:\nThere are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5.

Question:\nLeah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:\nOriginally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39.

Question:\nJason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:\nJason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8.

Question:\nShawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:\nShawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9.

Question:\nThere were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:\nThere were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. #### 29.

Question:\nMichael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:\nMichael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33.

Question:\nOlivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:\
"""
# Changed `The answer is` to `####`
    return fewshot_context


def gsm8k_fewshot_output(x):
    return "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. #### 8."

def gsm8k_input(x):
    return "Question: " + x["input"] + "\nAnswer:"

def gsm8k_output(x):
    answer = x["target"]
    return answer

def gsm8k_eval(prediction, ground_truth):
    try:
        prediction = str(prediction).replace(",", "")
        prediction = float(prediction)
        ground_truth = float(ground_truth)
        score = 1 if abs(prediction - ground_truth) < 1e-3 else 0
    except Exception as e:
        # print("Exception:", e)
        score = 0

    return score

GSMHardDataset = partial(
    TransformedDataset,
    data_path="reasoning-machines/gsm-hard",
    input_text=gsm8k_input,
    output_text=gsm8k_output,
    # fewshot_input_text=gsm8k_fewshot_input,
    # fewshot_output_text=gsm8k_fewshot_output,
    evaluation=gsm8k_eval,
    test_split="train",
    # fewshot_split="train",
)

GSMHardGenerateTestsDataset = partial(
    TransformedDataset,
    data_path="reasoning-machines/gsm-hard",
    input_text=lambda x: "Given a task, write tests that could verify if the assumptions made are correct"+x["input"],
    output_text=gsm8k_output,
    # fewshot_input_text=gsm8k_fewshot_input,
    # fewshot_output_text=gsm8k_fewshot_output,
    evaluation=gsm8k_eval,
    test_split="train",
    # fewshot_split="train",
)

if __name__ == "__main__":

    dataset = GSMHardDataset(
        num_fewshot=0,
        sampler=None,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)
