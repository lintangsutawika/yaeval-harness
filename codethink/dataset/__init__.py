from .data import TransformedDataset

from .svamp import SVAMPDataset
from .gsm8k import GSM8KDataset
from .aqua import AQUADataset
from .multiarith import MultiArithDataset
from .tabmwp import TabMWPDataset
from .bbh import BBHDataset
from .gsm_hard import GSMHardDataset
from .finqa import FinQADataset
from .mathqa import MathQADataset
from .coinflip import CoinFlipDataset
from .algebra import AlgebraDataset
from .cnn import CNNDataset, CNNQuestionDataset, qa_dataset
from .arc_challenge import ARCDataset

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
    "coinflip": CoinFlipDataset,
    "algebra": AlgebraDataset,
    "cnn_summarize": CNNDataset,
    "cnn_question": CNNQuestionDataset,
    **qa_dataset,
    "arc_challenge": ARCDataset,
}