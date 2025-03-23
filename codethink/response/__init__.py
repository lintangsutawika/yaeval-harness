from .code_responses import *
from .cot_responses import *
from .math_responses import *
from .routing import *

POSTPROCESS = {
    "code": execute_code,
    "cot": extract_answer,
    }
