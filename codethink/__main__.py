import os
import sys
import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

import argparse
import importlib.util


from codethink.utils import simple_parse_args_string
# from codethink import INTERFACE, SYSTEM_MESSAGE
# from codethink.dataset import DATASET as TASK_LIST
from codethink.evaluation import EvaluateSystem

logger = logging.getLogger(__name__)

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model_str", "-m", type=str, help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--inference_mode", "-i",
        type=str,
        default="default",
        help="Solve task by generating code or other test time inference approaches"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of inference run",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output file",
    )
    parser.add_argument(
        "--get_answer_expr",
        type=str,
        default="solution()",
        help="Name of function to execute",
    )
    parser.add_argument(
        "--get_answer_symbol",
        type=str,
        default=None,
        help="Postprocessing answer",
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
        "--revision", "-r",
        type=str,
        default="main",
        help="Set specific model version",
    )
    parser.add_argument(
        "--model_kwargs", "-a",
        default="",
        type=str,
        help="Comma separated string arguments for `from_pretrained`",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Set model in a particular device",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        type=str,
        help="Map model to device",
    )
    parser.add_argument(
        "--temperature",
        default=0.0,
        type=float,
        help="temperature",
    )
    parser.add_argument(
        "--top_p",
        default=1.0,
        type=float,
        help="top_p",
    )
    parser.add_argument(
        "--num_fewshot",
        default=0,
        type=int,
        help="Number of fewshot examples",
    )
    parser.add_argument(
        "--repeat",
        default=1,
        type=int,
        help="Number of repeats",
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="Number of seed",
    )
    parser.add_argument(
        "--max_model_len",
        default=4096,
        type=int,
        help="Max model lengths",
    )
    parser.add_argument(
        "--data_parallel_size",
        default=1,
        type=int,
        help="data parallel size",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        default=1,
        type=int,
        help="tensor parallel size",
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        help="Task to evaluate model on",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Rescore the output",
    )
    parser.add_argument(
        "--system_message",
        default=None,
        type=str,
        help="Custom system message",
    )
    parser.add_argument(
        "--use_system_role",
        action="store_true",
        help="Put System message in the system role (might not be available for all models)",
    )
    parser.add_argument(
        "--n_samples",
        default=None,
        type=str,
        help="Number of samples to infer on",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size",
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
        "--include_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--server_args",
        default=None,
        type=str,
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

    if args.serve:

        import requests

        def check_api_health(url):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return True
                else:
                    return False
            except requests.exceptions.RequestException:
                return False

        def launch_vllm_serve():
            # Construct the command to start the vLLM server
            command = ["vllm", "serve", args.model_str, "--disable-log-stats"] 
            if args.server_args:
                server_args_dict = simple_parse_args_string(args.server_args)
                command += [item for kv_pair in server_args_dict.items() for item in kv_pair]

            # Start the process
            process = subprocess.Popen(command)
            return process

        # Start the vLLM server
        vllm_server = launch_vllm_serve()

        url = "http://localhost:8000"
        while not check_api_health(url+"/health"):
            time.sleep(1)

    logger.info(f"Run: {run_name}")
    logger.info(
        "\n{} Run Configuration {}\n{}\n{}".format(
            "#"*33, "#"*33,
            "\n".join([" "*(32-len(key))+f"{key}: {value}" for key, value in vars(args).items()]),
            "#"*85,
            )
        )

    if args.trust_remote_code:
        logger.info(
            "Passed `--trust_remote_code`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`"
        )
        # Adopted from https://github.com/EleutherAI/lm-evaluation-harness.git
        # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
        # because it's already been determined based on the prior env var before launching our
        # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        # args.model_kwargs = args.model_kwargs + ",trust_remote_code=True"

    if args.include_path is not None:
        ADDITIONAL_TASK_LIST = dynamic_import("DATASET", args.include_path)
        ALL_TASK_LIST = {**ADDITIONAL_TASK_LIST, **TASK_LIST}
    else:
        ALL_TASK_LIST = TASK_LIST

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
            task_run_name = f"{args.model_str}-{task}-{args.system_message}"
        else:
            if len(task_list) > 1:
                task_run_name = f"{run_name}-{task}-{args.system_message}"
            else:
                task_run_name = run_name
        task_run_name = task_run_name.replace("/", "-")
        eval_dataset = ALL_TASK_LIST[task](
            num_fewshot=args.num_fewshot,
            sampler=None,
            n_samples=args.n_samples,
            data_kwargs=data_kwargs,
            **task_kwargs,
        )

        evaluator = EvaluateSystem(
            api_key=args.api_key,
            api_base=args.api_base,
            inference_kwargs=args.inference_kwargs,
            model_system=model_system,
            eval_dataset,
            run_name=task_run_name,
            output_path=args.output_path,
            run_args=vars(args),
            batch_size=args.batch_size,
            verbose=args.verbose,
            use_run_name=False if args.use_output_path_only == True else True,
        )

        evaluator.run(
            temperature=args.temperature,
            top_p=args.top_p,
            repeat=args.repeat,
            seed=args.seed
        )

    if args.serve:
        def kill_vllm_serve(process):
            process.terminate()
            process.wait()

        # Kill the vLLM server
        kill_vllm_serve(vllm_server)

if __name__ == "__main__":

    main()
