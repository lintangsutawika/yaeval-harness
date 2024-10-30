from .data import TransformedDataset

from .svamp import SVAMPDataset
from .gsm8k import GSM8KDataset
from .aqua import AQUADataset
from .multiarith import MultiArithDataset
from .tabmwp import TabMWPDataset

DATASET = {
    "svamp": SVAMPDataset,
    "gsm8k": GSM8KDataset,
    "aqua": AQUADataset,
    "multiarith": MultiArithDataset,
    "tabmwp": TabMWPDataset,
}