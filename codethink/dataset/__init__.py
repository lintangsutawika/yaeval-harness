import os
import glob
import importlib

from functools import partial
# from .data import TransformedDataset

TASK_LIST = {}

# Decorator to register functions
def register_task(name):
    def decorator(obj):
        obj = partial(obj, name=name)
        TASK_LIST[name] = obj
        globals()[name] = obj
        return obj
    return decorator

module_files = glob.glob(
    os.path.join(
        os.path.dirname(__file__), "*.py"
        )
    )

for file in module_files:
    module_name = os.path.basename(file)[:-3]
    if module_name != "__init__" and module_name.isidentifier():
        importlib.import_module(f".{module_name}", package=__name__)

__all__ = list(TASK_LIST.keys())

