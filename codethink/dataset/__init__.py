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
from .hendrycks_math import mathdatasets

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
    "lastletterconcat": LastLetterConcatDataset,
    "algebra": AlgebraDataset,
    "arc_challenge": ARCDataset,
    "aime": AIMEDataset,
    **mathdatasets,
}
