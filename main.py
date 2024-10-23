import re
import logging
import argparse

from tqdm import tqdm
from datasets import load_dataset

from codethink.utils import simple_parse_args_string
from codethink.interface import HFProgramInterface, HFNatLangInterface
from codethink.dataset import TransformedDataset
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
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser.add_argument(
        "--return_generation",
        action="store_true",
        help="Return system output",
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

    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

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

        args.model_kwargs = args.model_kwargs + ",trust_remote_code=True"

    if args.inference_mode == "code":
        num_fewshot = 0
        system_message = '''
            Write a function to solve a given problem by the user. Only write the program. Do not use `print`.
            The function must be named solution() and return `value` where value is only a number without any signs like '$' or '%'.
            '''
        model_system = HFProgramInterface(
            model=args.model_str,
            system_message=system_message,
            get_answer_expr=args.get_answer_expr,
            verbose=args.verbose,
            model_kwargs=simple_parse_args_string(args.model_kwargs)
            )

        gsm8k_fewshot_output = None
    else:
        num_fewshot = 0
        system_message = '''
            Solve the problem by thinking step-by-step. Go through the reasoning in order to derive the final answer.
            The final answer should follow the words 'So the answer is'.
            '''
        model_system = HFNatLangInterface(
            model=args.model_str,
            system_message=system_message,
            get_answer_symbol=r"the answer is (\-?[0-9\.\,]+)",
            verbose=args.verbose,
            model_kwargs=simple_parse_args_string(args.model_kwargs)
            )

        def gsm8k_fewshot_output(x):
            answer = x["answer"]
            answer = re.sub("####", "So the answer is", answer)
            return answer


    def gsm8k_output(x):
        answer = x["answer"]
        answer = answer.split("#### ")[-1]
        answer = float(re.findall(r'\d+', answer)[0])
        return answer

    gsm8k_dataset = TransformedDataset(
        data_path="gsm8k",
        data_name="main",
        input_text="question",
        output_text=gsm8k_output,
        fewshot_output_text=gsm8k_fewshot_output,
        test_split="test",
        fewshot_split="train",
        num_fewshot=num_fewshot,
        sampler=None,
    )

    evaluator = EvaluateSystem(
        model_system=model_system,
        dataset=gsm8k_dataset,
        return_generation=args.return_generation,
        run_name=args.run_name,
        output_path=args.output_path,
    )

    evaluator.run()
