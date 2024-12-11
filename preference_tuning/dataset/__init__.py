# from .routing_preference import code_or_natlang_paired_dataset
from .mathinstruct_preference import mathinstruct_preference, MathInstructPoTDataset, MathInstructCoTDataset

DATASET = {
    "mathinstrucy_pot": MathInstructPoTDataset,
    "mathinstrucy_cot": MathInstructCoTDataset,
}
