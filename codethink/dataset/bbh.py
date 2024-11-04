import os
from functools import partial

try:
    from codethink.dataset.data import TransformedDataset
except:
    from data import TransformedDataset

dir_path = os.path.dirname(os.path.realpath(__file__))

def bbh_input(x):
    return "Question:\n"+x["input"].strip()+"\nAnswer:"

def bbh_output(x):
    return x["target"]

# def bbh_fewshot_output(x):
#     return f"Let's think step by step. {x["solution"]} #### {x["answer"]}"

def convert_bool(x):
    if x == "True":
        return True
    elif x == "False":
        return False
    else:
        return x

def bbh_eval(prediction, ground_truth):

    ground_truth = convert_bool(ground_truth)
    prediction = convert_bool(prediction)

    try:
        prediction = float(prediction)
        ground_truth = float(ground_truth)
        score = 1 if abs(prediction - ground_truth) < 1e-3 else 0
    except Exception as e:
        prediction = str(prediction)
        ground_truth = str(ground_truth)
        score = 1 if prediction == ground_truth else 0

    return score


bbh_tasks = [
    'boolean_expressions',
	'causal_judgement',
	'date_understanding',
	'disambiguation_qa',
	'dyck_languages',
	'formal_fallacies',
	'geometric_shapes',
	'hyperbaton',
	'logical_deduction_five_objects',
	'logical_deduction_seven_objects',
	'logical_deduction_three_objects',
	'movie_recommendation',
	'multistep_arithmetic_two',
	'navigate',
	'object_counting',
	'penguins_in_a_table',
	'reasoning_about_colored_objects',
	'ruin_names',
	'salient_translation_error_detection',
	'snarks',
	'sports_understanding',
	'temporal_sequences',
	'tracking_shuffled_objects_five_objects',
	'tracking_shuffled_objects_seven_objects',
	'tracking_shuffled_objects_three_objects',
	'web_of_lies',
	'word_sorting',
]

BBHDataset = {}

for task in bbh_tasks:
    BBHDataset[task] = partial(
        TransformedDataset,
        data_path="lighteval/big_bench_hard",
        data_name=task,
        input_text=bbh_input,
        output_text=bbh_output,
        # fewshot_output_text=bbh_fewshot_output,
        eval=bbh_eval,
        test_split="train",
        # fewshot_split="dev",
    )

if __name__ == "__main__":

    dataset = BBHDataset["boolean_expressions"](
        num_fewshot=0,
        sampler=None,
        # trust_remote_code=True,
    )

    _input, _output = dataset.__getitem__(0)
    print("#### Input ###")
    print(_input)
    print("#### Output ###")
    print(_output)