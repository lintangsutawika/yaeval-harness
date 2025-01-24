import os

# Task
# Do initial inference of task
# Do parallel process?
# Route strings? i.e `next='task_name'`
# Merge and stuff

# Dataset
# Should fewshot be hardcoded?
# Should postprocessing change the dataset name? Or Task name?

# Build Instances, sample_id, task_id

class Task:
    def __init__(self,
                 name,
                 subtask_list: list = None,
                 dataset = None,
                 dataset_kwargs: dict = {},
                 preprocessor: callable = None,
                 postprocessor: callable = None,
                 inference_fn: callable = None,
                 ):
        self.name = name
        self.subtask_list = subtask_list
        self.dataset = dataset(**dataset_kwargs)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        if inference_fn is None:
            self.inference_fn = lambda x: x
        else:
            self.inference_fn = inference_fn

    def run(self, idx, task_id=None):
        state = {
            "task_id": task_id,
            "sample_id": idx,
            "current_step": 0,
            "step": []
            }

        if subtask_list is None:
            output, _state = self.run_task(idx, state)
            state["step"].append({"step_id": 0, "task": self.name, **_state})
            return output, state

        for _id, task in enumerate(subtask_list):
            state["current_step"] += 1
            output, state = task.run_task(idx,
                                          state=state,
                                          inference_fn=self.inference_fn
                                          )
            state["step"].append({"step_id": _id, "task": task.name, **_state})
        return output, state

    def preprocess(self, x, state=None):
        if self.preprocessor is not None:
            return self.preprocessor(x, state)
        else:
            return x

    def postprocess(self, x, state=None):
        if self.postprocessor is not None:
            return self.postprocessor(x, state)
        else:
            return x

    def build_message(self, x):
        message = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": x},
            ]
        return message

    def run_task(self, idx, state=None, inference_fn=None):
        # State is what?
        # evals, inputs, outputs 
        # Accumulate states
        previous_output = state["current_output"]
        
        x, y = self.dataset.__getitem__(idx)
        new_state["raw_input"] = x
        new_state["groud_truth"] = y
        x = self.preprocess(x, state)
        x = self.build_message(x)
        new_state["full_input"] = x
        if inference_fn is None:
            o = self.inference_fn(x)
        else:
            o = inference_fn(x)
        state["raw_output"] = o
        o = self.postprocess(o, state)
        state["output"] = o
        new_state["eval"] = self.eval(o, y)

        return output, new_state

if __name__ == "__main__":
    from codethink.datasets import GSM8KDataset

    def preprocess_PL_or_NL(x, state):
        current_step = state["current_step"]
        solve_with = state["step"][current_step-1]["output"]
        if solve_with == "programming language":
            return "Question: " + x["question"] + "\nAnswer:"
        elif solve_with == "natural language":
            return x["question"]
        return x


    task_1 = Task(
        name="gsm8k_routing",
        dataset=GSM8KDataset,
        dataset_kwargs={"num_fewshot": 0},
        )

    task_2 = Task(
        name="gsm8k_solve",
        dataset=GSM8KDataset,
        dataset_kwargs={"num_fewshot": 0},
        preprocessor=preprocess_PL_or_NL,
        )

    all_task = Task(
        name="gsm8k_pipeline",
        subtask_list=[
            task_1,
            task_2,
            ],
        )

    all_task.run(0)
