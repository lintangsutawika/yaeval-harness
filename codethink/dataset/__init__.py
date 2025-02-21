import os
import glob
import importlib

from functools import partial

TASK_LIST = {}

# Decorator to register functions
def register_task(name,
                  subtask_list=None,
                  dataset=None,
                  preprocessor=None,
                  postprocessor=None,
                  inference_fn=None,
                  system_message=None,
                  evaluation=None,
                  sampling_args=None,
                  ):
    def decorator(obj):
        obj = partial(obj,
                      name=name,
                      subtask_list=subtask_list,
                      dataset=dataset,
                      preprocessor=preprocessor,
                      postprocessor=postprocessor,
                      inference_fn=inference_fn,
                      system_message=system_message,
                      evaluation=evaluation,
                      sampling_args=sampling_args,
                      )
        TASK_LIST[name] = obj
        globals()[name] = obj
        return obj
    return decorator

def import_modules(path=None):

    if path is None:
        path = os.path.dirname(__file__)

    module_files = glob.glob(
        os.path.join(
            path, "**", "*.py"
            ), recursive=True
        )

    for file in module_files:
        module_name = os.path.basename(file)[:-3]
        if module_name != "__init__" and module_name.isidentifier():
            spec = importlib.util.spec_from_file_location(f"{module_name}", file)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            # importlib.import_module(f".{module_name}", package=__name__)

import_modules()

__all__ = list(TASK_LIST.keys())

