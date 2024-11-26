from .data import TransformedDataset

from .svamp import SVAMPDataset
from .gsm8k import GSM8KDataset
from .aqua import AQUADataset
from .multiarith import MultiArithDataset
from .tabmwp import TabMWPDataset
from .bbh import BBHDataset
from .gsm_hard import GSMHardDataset
from .finqa import FinQADataset
from .coinflip import CoinFlipDataset
from .algebra import AlgebraDataset
from .arc_challenge import ARCDataset
from .lastletterconcat import LastLetterConcatDataset
from .mathqa import MathQADataset
from .aime import AIMEDataset
from .routing_preference import code_or_natlang_paired_dataset

DATASET = {
    "svamp": SVAMPDataset,
    "gsm8k": GSM8KDataset,
    "aqua": AQUADataset,
    "multiarith": MultiArithDataset,
    "tabmwp": TabMWPDataset,
    **BBHDataset,
    "gsmhard": GSMHardDataset,
    "finqa": FinQADataset,
    "mathqa": MathQADataset,
    # "mathqa_numeric": MathQANumericAnswerDataset,
    # "mathqa_letter": MathQALetterAnswerDataset,
    "coinflip": CoinFlipDataset,
    "lastletterconcat": LastLetterConcatDataset,
    "algebra": AlgebraDataset,
    # "cnn_summarize": CNNDataset,
    # "cnn_question": CNNQuestionDataset,
    "arc_challenge": ARCDataset,
    "aime": AIMEDataset,
}