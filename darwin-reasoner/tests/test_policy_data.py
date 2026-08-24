from darwin_reasoner.training.policy_data import CausalSFTCollator


def test_collator_masks_and_pads():
    collator = CausalSFTCollator(pad_token_id=0)
    batch = collator([
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
    ])
    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert batch["input_ids"][1, 2].item() == 0
    assert batch["labels"][1, 2].item() == -100
