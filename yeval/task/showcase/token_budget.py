import re
from functools import partial
from yeval.log.usage import log_token_usage
from yeval.task import register_task, YevalTask

def exit_after_budget_exhausted(x, state, budget=1024):
    total_output_tokens = sum([step['log']['output_tokens'] for step in state['step']])
    if total_output_tokens >= budget:
        return True
    else:
        return False

@register_task("with_budget")
class WithBudgetTask(YevalTask):
    loop_exit=exit_after_budget_exhausted
    logging=log_token_usage
    loop_max=10
    sampling_args={
        "n":2
    }

