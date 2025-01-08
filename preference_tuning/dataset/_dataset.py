from typing import Any, Callable, Dict, List, Mapping, Optional

from torchtune.data import Message
from torchtune.datasets._preference import PreferenceDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

class PairedMessages(Transform):

    def __init__(
        self, train_on_input: bool = False, column_map: Optional[Dict[str, str]] = None
    ):
        self.train_on_input = train_on_input
        self._column_map = column_map

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        column_map = self._column_map or {}
        key_prompt = column_map.get("prompt", "prompt")
        key_chosen = column_map.get("chosen", "chosen")
        key_rejected = column_map.get("rejected", "rejected")

        chosen_messages = [
            Message(
                role="user", content=sample[key_prompt], masked=not self.train_on_input
            ),
            Message(role="assistant", content=sample[key_chosen]),
        ]

        rejected_messages = [
            Message(
                role="user", content=sample[key_prompt], masked=not self.train_on_input
            ),
            Message(role="assistant", content=sample[key_rejected]),
        ]

        return {"chosen": chosen_messages, "rejected": rejected_messages}

def dpo_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> PreferenceDataset:

    message_transform = PairedMessages(
        train_on_input=train_on_input, column_map=column_map
    )

    return PreferenceDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
