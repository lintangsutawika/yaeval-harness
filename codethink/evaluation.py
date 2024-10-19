from tqdm import tqdm


class EvaluateSystem:
    def __init__(self, dataset, model_system, verbose=False, run_name=None, output_file=None, **kwargs):
        self.dataset = dataset
        self.model_system = model_system
        self.verbose = verbose
        self.output_file = output_file

    def run(self, temperature=0.1):
        output_dict = {
            "idx": [],
            "score": [],
            "answer": [],
            "ground_truth": [],
            "system_output": [],
            "user_input": [],
        }
        all_scores = []
        for idx, sample in enumerate(tqdm(self.dataset)):
            user_input, ground_truth = sample
            system_output = None
            try:
                ans = self.model_system.run(user_input, temperature=temperature, return_generation=self.verbose)
                if self.verbose:
                    ans, system_output = ans
                ans = float(ans)
                score = 1 if abs(ans - ground_truth) < 1e-3 else 0
            except Exception as e:
                print("Exception:", e)
                ans = ''
                score = 0

            all_scores.append(score)
            if self.verbose:
                output_dict["idx"].append(idx)
                output_dict["score"].append(score)
                output_dict["answer"].append(ans)
                output_dict["ground_truth"].append(ground_truth)
                output_dict["system_output"].append(system_output)            
                output_dict["user_input"].append(user_input)

        print(f"Score: {sum(all_scores)/len(all_scores)}")
        # with open(self.output_file, "r") as file:
        #     for i in range(len(self.dataset)):
        #         file.write()
            