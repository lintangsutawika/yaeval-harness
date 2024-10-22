import os
import logging
import datetime
import jsonlines

from tqdm import tqdm

logger = logging.getLogger(__name__)

class EvaluateSystem:
    def __init__(self,
                 dataset,
                 model_system,
                 return_generation=False,
                 run_name=None,
                 output_path=None,
                 **kwargs
                 ):
        self.dataset = dataset
        self.model_system = model_system
        self.return_generation = return_generation
        if run_name is None:
            current_time = datetime.datetime.now()
            self.run_name = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        else:
            self.run_name = run_name
        self.output_path = output_path

    def run(self, temperature=0.1):
        all_scores = []
        output_json = []
        for idx, sample in tqdm(enumerate(self.dataset)):
            user_input, ground_truth = sample
            system_output = None
            try:
                ans = self.model_system.run(user_input, temperature=temperature, return_generation=self.return_generation)
                if self.return_generation:
                    ans, flops, system_output = ans
                else:
                    ans, flops = ans
                ans = float(ans)
                score = 1 if abs(ans - ground_truth) < 1e-3 else 0
                logger.info(f"Score: {score}, Flops: {flops}, Prediction: {ans}, Ground Truth: {ground_truth}")
            except Exception as e:
                print("Exception:", e)
                ans = ''
                score = 0

            all_scores.append(score)
            output_json.append(
                {
                    "idx": idx,
                    "score": score,
                    "answer": ans,
                    "flops": flops,
                    "ground_truth": ground_truth,
                    "system_output": system_output,
                    "user_input": user_input,
                }
            )

        logger.info(f"{self.run_name} complete")
        logger.info(f"Score on {self.dataset.name}: {sum(all_scores)/len(all_scores)}")
        if self.output_path is not None:
            os.path.join(self.output_path, f"{self.run_name}.jsonl")
            with jsonlines.open(self.output_file, "w") as file:
                file.write_all(output_json)
                
