import os
import json
import logging
import datetime
import jsonlines

from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import DataLoader

from codethink.utils import zeno_upload

logger = logging.getLogger(__name__)


class EvaluateSystem:
    def __init__(self,
                 dataset,
                 model_system,
                 run_name=None,
                 output_path=None,
                 run_args=None,
                 use_run_name=True,
                 batch_size=1,
                 verbose=False,
                 **kwargs
                 ):
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size)
        self.model_system = model_system
        if run_name is None:
            current_time = datetime.datetime.now()
            self.run_name = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        else:
            self.run_name = run_name
        self.output_path = output_path
        self.use_run_name = use_run_name
        self.run_args = run_args
        self.verbose = verbose

    def run(self, temperature=0.1, top_p=1.0, repeat=1, seed=None):
        result_dict = {
            "n_samples": 0,
            "duration": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        data_dict = {
            "idx": [],
            "answer": [],
            "system_output": [],
            "ground_truth": [],
            "user_input": [],
            "score": [],
            "duration": [],
            "input_tokens": [],
            "output_tokens": [],
            "total_tokens": [],
        }
        output_json = []
        idx = 0

        user_input = []
        ground_truth = []
        for inp, out in self.dataset:
            user_input.append(inp)
            ground_truth.append(out)

        ans_list, output_dict_list = self.model_system.run(user_input, temperature=temperature, top_p=top_p, repeat=repeat, seed=seed)

        for ans, gt, inp, output_dict in tqdm(zip(ans_list, ground_truth, user_input, output_dict_list), total=len(ans_list)):
            ans = self.dataset.extract_answer(ans)
            score_dict = self.dataset.eval(ans, gt)
            score_string = ", ".join([f"{metric}: {score}" for metric,score in score_dict.items()])

            if self.verbose:
                logger.info(f"\nId: {idx}, {score_string}, Prediction: {ans}, Ground Truth: {gt}")
            result_dict["n_samples"] += 1
            for metric, score in score_dict.items():
                if metric not in result_dict:
                    result_dict[metric] = score
                else:
                    result_dict[metric] += score

            result_dict["duration"] += output_dict["duration"]
            result_dict["input_tokens"] += output_dict["input_len"]
            result_dict["output_tokens"] += output_dict["output_len"]
            result_dict["total_tokens"] += (output_dict["input_len"] + output_dict["output_len"])
            output_json.append(
                {
                    "idx": idx,
                    **score_dict,
                    "ground_truth": gt,
                    "answer": ans,
                    "user_input": inp,
                    **output_dict,
                }
            )

            # data_dict["idx"].append(int(idx))
            # data_dict["answer"].append(ans)
            # data_dict["system_output"].append(output_dict["system_output"])
            # data_dict["ground_truth"].append(gt)
            # data_dict["user_input"].append(user_input)
            # data_dict["score"].append(score)
            # data_dict["duration"].append(output_dict["duration"])
            # data_dict["input_tokens"].append(output_dict["input_len"])
            # data_dict["output_tokens"].append(output_dict["output_len"])
            # data_dict["total_tokens"].append(output_dict["input_len"] + output_dict["output_len"])
            idx += 1
        # zeno_upload(self.run_name, data_dict)

        result_dict = {**self.run_args, **result_dict}
        for metric, score in score_dict.items():
            result_dict[f"avg_{metric}"] = result_dict[metric]/result_dict["n_samples"]
        result_dict["avg_duration"] = result_dict["duration"]/result_dict["n_samples"]
        result_dict["avg_input_tokens"] = result_dict["input_tokens"]/result_dict["n_samples"]
        result_dict["avg_output_tokens"] = result_dict["output_tokens"]/result_dict["n_samples"]
        result_dict["avg_total_tokens"] = result_dict["total_tokens"]/result_dict["n_samples"]
        logger.info(f"{self.run_name} complete")
        logger.info(
            ",".join(
                [f"{metric}: {result_dict[f'avg_{metric}']}" for metric in score_dict.keys()]
                )
            )

        if self.output_path is not None:
            if self.use_run_name:
                run_path = os.path.join(self.output_path, self.run_name)
            else:
                run_path = os.path.join(self.output_path)
            os.makedirs(run_path, exist_ok=True)
            result_file = os.path.join(run_path, "result.json")
            with open(result_file, 'w', encoding='utf-8') as file:
                json.dump(result_dict, file, ensure_ascii=False, indent=4)

            try:
                output_file = os.path.join(run_path, "output.jsonl")
                with jsonlines.open(output_file, "w") as file:
                    file.write_all(output_json)
            except Exception as e:
                print(e)

        return 0

class OracleEvaluation(EvaluateSystem):

    def run(self, temperature=0.1, top_p=1.0, repeat=1, seed=None):

        result_dict = {
            "n_samples": 0,
            "score": 0,
            "duration": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        output_json = []
        idx = 0

        while True:
            for sample in tqdm(self.data_loader):
                user_input, ground_truth = sample

                ans_list, output_dict_list = self.model_system.run(user_input, temperature=temperature, top_p=top_p, repeat=repeat, seed=seed)

                for ans, gt, inp, output_dict in zip(ans_list, ground_truth, user_input, output_dict_list):
                    score = self.dataset.eval(ans, gt)

                    if float(score) != 1.0:
                        ans, gt, inp, 

                    # if self.verbose:
                    logger.info(f"\nId: {idx}, Score: {score}, Prediction: {ans}, Ground Truth: {gt}")
                    result_dict["n_samples"] += 1
                    result_dict["score"] += score
                    result_dict["duration"] += output_dict["duration"]
                    result_dict["input_tokens"] += output_dict["input_len"]
                    result_dict["output_tokens"] += output_dict["output_len"]
                    result_dict["total_tokens"] += (output_dict["input_len"] + output_dict["output_len"])
                    output_json.append(
                        {
                            "idx": idx,
                            "score": score,
                            "ground_truth": gt,
                            "answer": ans,
                            "user_input": inp,
                            **output_dict,
                        }
                    )

        result_dict = {**self.run_args, **result_dict}
        result_dict["avg_score"] = result_dict["score"]/result_dict["n_samples"]
        result_dict["avg_duration"] = result_dict["duration"]/result_dict["n_samples"]
        result_dict["avg_input_tokens"] = result_dict["input_tokens"]/result_dict["n_samples"]
        result_dict["avg_output_tokens"] = result_dict["output_tokens"]/result_dict["n_samples"]
        result_dict["avg_total_tokens"] = result_dict["total_tokens"]/result_dict["n_samples"]
        logger.info(f"{self.run_name} complete")
        logger.info("Score: {}".format(result_dict["avg_score"]))

        if self.output_path is not None:
            run_path = os.path.join(self.output_path, self.run_name)
            os.makedirs(run_path, exist_ok=True)
            result_file = os.path.join(run_path, "result.json")
            with open(result_file, 'w', encoding='utf-8') as file:
                json.dump(result_dict, file, ensure_ascii=False, indent=4)

            try:
                output_file = os.path.join(run_path, "output.jsonl")
                with jsonlines.open(output_file, "w") as file:
                    file.write_all(output_json)
            except Exception as e:
                print(e)

        return 0
