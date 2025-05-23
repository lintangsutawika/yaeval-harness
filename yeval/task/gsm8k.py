import re
from functools import partial
from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

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
    return fewshot_context

def gsm8k_fewshot_output(x):
    return """\
Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. #### 8.\
"""

def gsm8k_input(x):
    return f"Question:\n{x['question']}\nAnswer:"

def gsm8k_output(x):
    answer = x["answer"]
    answer = answer.split("#### ")[-1]
    answer = re.findall(r'\d+', answer)[0]
    return answer

@register_task("gsm8k")
class GSM8KTask(YevalTask):
    data_path="gsm8k"
    data_name="main"
    input_text=gsm8k_input
    output_text=gsm8k_output
    # fewshot_input_text=gsm8k_fewshot_input
    # fewshot_output_text=gsm8k_fewshot_output
    # fewshot_split="train"
    test_split="test"
    evaluation={"accuracy": math_eval}
