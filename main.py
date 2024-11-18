import re
import logging
import argparse

from tqdm import tqdm
from datasets import load_dataset

from codethink.utils import simple_parse_args_string
# from codethink.interface import HFProgramInterface, HFNatLangInterface
from codethink import INTERFACE, SYSTEM_MESSAGE
from codethink.dataset import DATASET
from codethink.evaluation import EvaluateSystem


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model_str", "-m", type=str, help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--inference_mode", "-i", type=str, default="code", help="Solve task by generating code or other test time inference approaches"
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
        "--task",
        default="gsm8k",
        type=str,
        help="Task to evaluate model on",
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
        type=int,
        help="Number of samples to infer on",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    # parser.add_argument(
    #     "--alt_prompt",
    #     action="store_true",
    # )
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

if __name__ == "__main__":
    parser = setup_parser()
    args = parse_eval_args(parser)

    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    run_name = args.run_name.replace("/", "-")
    logger.info(f"Run: {run_name}")

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

    if args.system_message is not None:
        if args.system_message in SYSTEM_MESSAGE:
            system_message = SYSTEM_MESSAGE[args.system_message]
        else:
            system_message = args.system_message
    else:
        system_message = SYSTEM_MESSAGE[args.inference_mode]

    model_system = INTERFACE[args.inference_mode](
        model=args.model_str,
        system_message=system_message,
        get_answer_expr=args.get_answer_expr,
        verbose=args.verbose,
        use_system_role=args.use_system_role,
        trust_remote_code=args.trust_remote_code,
        # model_kwargs=simple_parse_args_string(args.model_kwargs),
        )

    eval_dataset = DATASET[args.task](
        num_fewshot=args.num_fewshot,
        sampler=None,
        n_samples=args.n_samples,
    )

    evaluator = EvaluateSystem(
        model_system=model_system,
        dataset=eval_dataset,
        run_name=run_name,
        output_path=args.output_path,
    )

    evaluator.run(temperature=args.temperature, top_p=args.top_p, repeat=args.repeat, seed=args.seed)
