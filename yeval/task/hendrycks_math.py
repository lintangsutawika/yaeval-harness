import re
from functools import partial

from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage
from yeval.metrics import math_eval
from yeval.response.math_responses import (
        get_boxed_answer
        )

from datasets import concatenate_datasets

def math_merge(dataset):
    return concatenate_datasets(concatenate_datasets)

def math_level_5(dataset):
    return dataset.filter(lambda x: x["level"] == "Level 5")

def math_input(x):
    return x['problem']

def math_output(x):
    return get_boxed_answer(x["solution"])

class MATHBaseTask(YevalTask):
    data_path="EleutherAI/hendrycks_math"
    input_text=math_input
    output_text=math_output
    preprocessing=math_level_5
    test_split="test"
    evaluation={"accuracy": math_eval}
    logging=log_token_usage

@register_task("math_algebra")
class MATHAlgebraTask(MATHBaseTask):
    data_name='algebra'

@register_task("math_counting_and_probability")
class MATHCountingAndProbabilityTask(MATHBaseTask):
    data_name='counting_and_probability'

@register_task("math_geometry")
class MATHGeometryTask(MATHBaseTask):
    data_name='geometry'

@register_task("math_intermediate_algebra")
class MATHIntermediateAlgebraTask(MATHBaseTask):
    data_name='intermediate_algebra'

@register_task("math_number_theory")
class MATHNumberTheoryTask(MATHBaseTask):
    data_name='number_theory'

@register_task("math_prealgebra")
class MATHPrealgebraTask(MATHBaseTask):
    data_name='prealgebra'

@register_task("math_precalculus")
class MATHPrecalculusTask(MATHBaseTask):
    data_name='precalculus'

@register_task("full_math_algebra")
class FullMATHAlgebraTask(MATHBaseTask):
    data_name='algebra'
    preprocessing=None

@register_task("full_math_counting_and_probability")
class FullMATHCountingAndProbabilityTask(FullMATHAlgebraTask):
    data_name='counting_and_probability'

@register_task("full_math_geometry")
class FullMATHGeometryTask(FullMATHAlgebraTask):
    data_name='geometry'

@register_task("full_math_intermediate_algebra")
class FullMATHIntermediateAlgebraTask(FullMATHAlgebraTask):
    data_name='intermediate_algebra'

@register_task("full_math_number_theory")
class FullMATHNumberTheoryTask(FullMATHAlgebraTask):
    data_name='number_theory'

@register_task("full_math_prealgebra")
class FullMATHPrealgebraTask(FullMATHAlgebraTask):
    data_name='prealgebra'

@register_task("full_math_precalculus")
class FullMATHPrecalculusTask(FullMATHAlgebraTask):
    data_name='precalculus'
