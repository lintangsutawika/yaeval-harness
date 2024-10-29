import os
import json
import logging
import datetime
import jsonlines

from tqdm import tqdm

from typing import List, Tuple

logger = logging.getLogger(__name__)


class EvaluateSystem:
    def __init__(self,
                 dataset,
                 model_system,
                 run_name=None,
                 output_path=None,
                 **kwargs
                 ):
        self.dataset = dataset
        self.model_system = model_system
        if run_name is None:
            current_time = datetime.datetime.now()
            self.run_name = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        else:
            self.run_name = run_name
        self.output_path = output_path

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
        for idx, sample in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            user_input, ground_truth = sample

            ans, output_dict = self.model_system.run(user_input, temperature=temperature, top_p=top_p, repeat=repeat, seed=seed)

            try:
                ans = str(ans).replace(",", "")
                ans = float(ans)
                ground_truth = float(ground_truth)
                score = 1 if abs(ans - ground_truth) < 1e-3 else 0
            except Exception as e:
                print("Exception:", e)
                ans = ''
                score = 0

            logger.info(f"Score: {score}, Prediction: {ans}, Ground Truth: {ground_truth}")
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
                    "answer": ans,
                    "ground_truth": ground_truth,
                    "user_input": user_input,
                    **output_dict,
                }
            )

        result_dict["avg_score"] = result_dict["score"]/result_dict["n_samples"]
        result_dict["avg_duration"] = result_dict["duration"]/result_dict["n_samples"]
        result_dict["avg_input_tokens"] = result_dict["input_tokens"]/result_dict["n_samples"]
        result_dict["avg_output_tokens"] = result_dict["output_tokens"]/result_dict["n_samples"]
        result_dict["avg_total_tokens"] = result_dict["total_tokens"]/result_dict["n_samples"]
        logger.info(f"{self.run_name} complete")
        logger.info("Score: {}".format(result_dict["avg_score"]))

        run_path = os.path.join(self.output_path, self.run_name)
        os.makedirs(run_path, exist_ok=True)
        if self.output_path is not None:
            output_file = os.path.join(run_path, "output.jsonl")
            with jsonlines.open(output_file, "w") as file:
                file.write_all(output_json)

            result_file = os.path.join(run_path, "result.json")
            with open(result_file, 'w', encoding='utf-8') as file:
                json.dump(result_dict, file, ensure_ascii=False, indent=4)

        return 0