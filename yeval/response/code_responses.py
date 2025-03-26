import sys
import subprocess

def process_generation_to_code(gens: str, answer_expr: str):
    if '```python' in gens:
        gens = gens.split('```python')[1].split('```')[0]
    elif '```' in gens:
        gens = gens.split('```')[1].split('```')[0]
    elif answer_expr in gens:
        gens = "def "+answer_expr+f"{answer_expr}".join(gens.split(answer_expr)[1:])
    else:
        return False
        
    return gens.split('\n')

def is_runnable_code(text_string, answer_expr='solution()', time_out=10):
    # Check if the _output is a program
    code = process_generation_to_code(text_string, answer_expr)
    if code:
        def _generate_code(code, answer_expr):
            return "\n".join(code)+f"\nans = 'ans='+str({answer_expr})\nprint(ans)"
        # Generate code snippet that will be executed in a different process
        code_snippet = _generate_code(code, answer_expr)
        try:
            subprocess_result = subprocess.run([sys.executable, "-c", code_snippet], timeout=time_out, text=True, capture_output=True)
            exec_result = subprocess_result.stdout.split("ans=")[-1].strip()
            return exec_result
        except Exception as e:
            return False
    else:
        return False

def execute_code(x):
    exec_result = is_runnable_code(x) 
    if exec_result:
        return exec_result
    return x
