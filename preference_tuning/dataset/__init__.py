from ._dataset import dpo_dataset
from .mathinstruct_preference import MathInstructPoTDataset, MathInstructCoTDataset

DATASET = {
    "mathinstruct_pot": MathInstructPoTDataset,
    "mathinstruct_cot": MathInstructCoTDataset,
}
