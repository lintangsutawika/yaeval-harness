import re
from functools import partial
from yeval.task import register_task, YevalTask

@register_task("with_budget")
class WithBudgetTask(YevalTask):
    exit_loop=exit_after_budget_exhausted

def exit_after_budget_exhausted(x, state):
    print("state", state)
    return True