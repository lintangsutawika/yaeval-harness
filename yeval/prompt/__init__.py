import os
import glob
import importlib
from functools import partial

from .base_prompt import YevalPrompt
from yeval.response import get_postprocess_fn

PROMPT_LIST = {}

def get_message_str(system_message):
    if system_message is None:
        return None

    if system_message in PROMPT_LIST:
        system_message = PROMPT_LIST[system_message]

    if isinstance(system_message, str):
        return system_message
    else:
        if not issubclass(system_message, YevalPrompt):
            raise TypeError(f"Expected a YevalPrompt subclass, got {type(system_message)}")
        else:
            return system_message.system_message

def get_prompt(prompt):
    if prompt is None:
        return None, None
    postprocessor = None
    if prompt in PROMPT_LIST:
        prompt = PROMPT_LIST[prompt]

    if issubclass(prompt, YevalPrompt):
        system_message = getattr(prompt, "system_message")
        postprocessor = getattr(prompt, "postprocessor")
    else:
        system_message = system_message
        postprocessor = None

    if postprocessor is not None:
        postprocessor = get_postprocess_fn(postprocessor)
    return system_message, postprocessor


# Decorator to register functions
def register_prompt(name):
    def decorator(obj):
        PROMPT_LIST[name] = obj
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

__all__ = list(PROMPT_LIST.keys())

