from .code_responses import *
from .cot_responses import *
from .math_responses import *
from .routing import *

POSTPROCESS = {
    "code": execute_code,
    "cot": extract_answer,
    "box": get_boxed_answer,
    }

def get_postprocess_fn(postprocess):
    if postprocess in POSTPROCESS:
        return POSTPROCESS[postprocess]
    else:
        return postprocess
