from .data import TransformedDataset

from .svamp import SVAMPDataset
from .gsm8k import GSM8KDataset


DATASET = {
    "svamp": SVAMPDataset,
    "gsm8k": GSM8KDataset,
}