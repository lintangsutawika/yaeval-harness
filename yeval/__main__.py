import os
import sys
import atexit
import asyncio
import logging
import requests
import subprocess
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)

import argparse
import importlib.util

from yeval.utils import simple_parse_args_string

from yeval.task import TASK_LIST
from yeval.prompt import PROMPT_LIST
from yeval.response import POSTPROCESS
from yeval.evaluation import EvaluateSystem

logger = logging.getLogger(__name__)

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model", "-m", type=str, help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of inference run",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output file",
    )
    parser.add_argument(
        "--serve", "-s",
        action="store_true",
        help="Serve model while also running evaluation",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Sets verbose",
    )
    parser.add_argument(
        "--num_fewshot",
        default=0,
        type=int,
        help="Number of fewshot examples",
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="Number of seed",
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        help="Task to evaluate model on",
    )
    parser.add_argument(
        "--post",
        default=None,
        type=str,
        help="Post Process",
    )
    # parser.add_argument(
    #     "--rescore",
    #     action="store_true",
    #     help="Rescore the output",
    # )
    parser.add_argument(
        "--system_message",
        default=None,
        type=str,
        help="Custom system message",
    )
    parser.add_argument(
        "--use_output_path_only",
        action="store_true",
        help="directly save output to output path",
    )
    parser.add_argument(
        "--data_kwargs",
        default=None,
        type=str,
        help="Data key args",
    )
    parser.add_argument(
        "--task_kwargs",
        default=None,
        type=str,
        help="task related args",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the server running",
    )
    parser.add_argument(
        "--n_samples",
        default=None,
        type=int,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--include_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--server_args",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--sample_args",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--no_system_role",
        action="store_true",
    )
    return parser

def parse_eval_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(
                    f"Argument '{action.dest}' doesn't have a type specified."
                )
            else:
                continue
    return parser.parse_args()

def load_script(module_name, script_path):
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Script not found at {script_path}")

    module_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def dynamic_import(module_name, script_path):
    abs_script_path = os.path.abspath(script_path)

    if os.path.isdir(abs_script_path) and os.path.isfile(os.path.join(abs_script_path, "__init__.py")):
        package_name = os.path.basename(abs_script_path)
        parent_dir = os.path.dirname(abs_script_path)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        package = importlib.import_module(package_name)

    if not hasattr(package, module_name):
        raise AttributeError(f"Module '{module_name}' not found in package '{package_name}'.")
    
    return getattr(package, module_name)

def main():
    parser = setup_parser()
    args = parse_eval_args(parser)

    run_name = args.run_name

    api_key = args.api_key or "EMPTY"
    api_base = args.api_base or "http://localhost:8000/v1"

    if args.serve:
        import time
        import requests
        import subprocess
        from yeval.utils import check_api_health

        def launch_vllm_serve():
            # Construct the command to start the vLLM server
            command = ["vllm", "serve", args.model, "--disable-log-stats"]
            if args.server_args:
                server_args_dict = simple_parse_args_string(args.server_args)
                server_args_dict = {f"--{k}":v for k,v in server_args_dict.items() if v is not None}
                if "max_model_len" not in server_args_dict:
                    server_args_dict["--max_model_len"] = 4096
                    server_args_dict["--enable-chunked-prefill"] = False
                command += [str(item) for kv_pair in server_args_dict.items() for item in kv_pair]

            # Start the process
            with open("serve.log", "w") as log_file:
                process = subprocess.Popen(
                    command,
                    env={**os.environ, "VLLM_CONFIGURE_LOGGING": "0"},
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

            return process

        def kill_vllm_serve(process):
            process.terminate()
            process.wait()

        def on_exit(args, process):
            # Kill the vLLM server
            if args.serve and (args.keep == False):
                logging.info("Killing vLLM server")
                kill_vllm_serve(process)

        # Start the vLLM server
        vllm_server = launch_vllm_serve()
        atexit.register(
            on_exit,
            args=args, process=vllm_server
        )

        url = "http://localhost:8000"
        while not check_api_health(url.split("/v1")[0]+"/health"):
            time.sleep(1)

    logger.warning(f"Run: {run_name}")
    logger.warning(
        "\n{} Run Configuration {}\n{}\n{}".format(
            "#"*33, "#"*33,
            "\n".join([" "*(32-len(key))+f"{key}: {value}" for key, value in vars(args).items()]),
            "#"*85,
            )
        )

    if args.trust_remote_code:
        logger.warning(
            "Passed `--trust_remote_code`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`"
        )
        # Adopted from https://github.com/EleutherAI/lm-evaluation-harness.git
        # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
        # because it's already been determined based on the prior env var before launching our
        # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        # args.model_kwargs = args.model_kwargs + ",trust_remote_code=True"
    aux_task_args = {}
    if args.no_system_role:
        aux_task_args["system_role"] = False

    if args.post and (args.post in POSTPROCESS):
        aux_task_args["postprocessor"] = POSTPROCESS[args.post]

    evaluator = EvaluateSystem(
        model=args.model,
        api_key=api_key,
        api_base=api_base,
        system_message=args.system_message,
        output_path=args.output_path,
        use_run_name=~args.use_output_path_only
        )

    if args.include_path is not None:
        from yeval.task import import_modules
        logger.warning(f"Importing modules from {args.include_path}")
        import_modules(args.include_path)

    if args.data_kwargs is not None:
        data_kwargs = eval(args.data_kwargs)
    else:
        data_kwargs = None

    if args.task_kwargs is not None:
        task_kwargs = eval(args.task_kwargs)
    else:
        task_kwargs = {}

    task_list = args.task.split(",")
    for task in task_list:
        logger.info(f"Task: {task}")
        if run_name is None:
            task_run_name = f"{args.model}-{task}-{args.system_message}"
        else:
            if len(task_list) > 1:
                task_run_name = f"{run_name}-{task}-{args.system_message}"
            else:
                task_run_name = run_name
        task_run_name = task_run_name.replace("/", "-")
        
        asyncio.run(
            evaluator.run(
            TASK_LIST[task](name=task, **aux_task_args),
            sampling_args=simple_parse_args_string(args.sample_args) if args.sample_args else None,
            run_name=task_run_name,
            n_samples=args.n_samples
        ))


if __name__ == "__main__":
    # fire.Fire(main)
    main()
