from ._dataset import dpo_dataset
from .mathinstruct_preference import MathInstructPoTDataset, MathInstructCoTDataset

DATASET = {
    "mathinstrucy_pot": MathInstructPoTDataset,
    "mathinstrucy_cot": MathInstructCoTDataset,
}
