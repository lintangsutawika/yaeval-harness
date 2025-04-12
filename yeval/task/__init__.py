import os
import glob
import importlib
import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)

from functools import partial

from yeval.utils import import_modules

from .base_task import YevalTask
from .base_data import YevalDataset

TASK_LIST = {}

# Decorator to register functions
def register_task(name,
                  #subtask_list=None,
                  #dataset=None,
                  #preprocessor=None,
                  #postprocessor=None,
                  #inference_fn=None,
                  #system_message=None,
                  #evaluation=None,
                  #sampling_args=None,
                  #logging=None,
                  ):
    def decorator(obj):
        #obj = partial(obj,
        #              name=name,
        #              subtask_list=subtask_list,
        #              dataset=dataset,
        #              preprocessor=preprocessor,
        #              postprocessor=postprocessor,
        #              inference_fn=inference_fn,
        #              system_message=system_message,
        #              evaluation=evaluation,
        #              sampling_args=sampling_args,
        #              logging=logging,
        #              )
        TASK_LIST[name] = obj
        globals()[name] = obj
        return obj
    return decorator

path = os.path.dirname(__file__)
import_modules(path)

__all__ = list(TASK_LIST.keys())

