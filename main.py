import re
import argparse

from tqdm import tqdm
from datasets import load_dataset

from codethink.interface import HFProgramInterface
from codethink.evaluation import EvaluateSystem


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model_str", "-m", type=str, help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--inference_mode", "-i", type=str, help="Solve task by generating code or other test time inference approaches"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
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

    if args.inference_mode == "code":
        model = HFProgramInterface(
            args.model_str,
            get_answer_expr=args.get_answer_expr,
            verbose=args.verbose,
            **{"trust_remote_code": self.trust_remote_code}
            )
    
    evaluator = EvaluateSystem(
        model=model,
        # dataset=dataset
    )