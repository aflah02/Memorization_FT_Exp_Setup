import numpy as np
import os
from datasets import Dataset, DatasetDict, load_dataset
import string
import torch
from transformers import PreTrainedTokenizer
import logging

def generate_random_string_dataset(seed: int, num_sequences: int, sequence_length: int):
    rng = np.random.default_rng(seed)
    sequences = generate_strings(
        num_sequences, sequence_length, rng
    )
    dataset = Dataset.from_dict(
        {
            "text": sequences,
        }
    )
    datasets = DatasetDict(
        {
            "train": dataset,
            "test": dataset,
        }
    )
    datasets.set_format("torch")
    return datasets

def generate_strings(num_sequences: int, sequence_length: int, rng):
    letters = string.ascii_lowercase
    sequences = rng.choice(list(letters), size=(num_sequences, sequence_length))
    return ["".join(seq) for seq in sequences]

def encode_character_wise(tokenizer: PreTrainedTokenizer, dataset: Dataset):
    def characterwise_encoding(example):
        sequences = example["text"]
        max_length = max(len(s) for s in sequences)
        sequence_token_ids = []
        sequence_token_masks = []
        for sequence in sequences:
            sequence_chars = list(sequence)
            encoded_chars = tokenize(
                tokenizer,
                sequence_chars,
                max_length=1,
                logger=logging.getLogger(__name__),
            )
            # add padding
            num_padding = max_length - len(sequence)
            padded_input_ids = torch.cat(
                (
                    torch.tensor(
                        [tokenizer.pad_token_id] * num_padding, dtype=torch.long
                    ),
                    encoded_chars.input_ids.squeeze(1),
                )
            )
            padded_attention_mask = torch.cat(
                (
                    torch.tensor([0] * num_padding, dtype=torch.long),
                    encoded_chars.attention_mask.squeeze(1),
                )
            )
            sequence_token_ids.append(padded_input_ids)
            sequence_token_masks.append(padded_attention_mask)
        return {
            "input_ids": torch.stack(sequence_token_ids),
            "attention_mask": torch.stack(sequence_token_masks),
        }

    return dataset.map(
        characterwise_encoding,
        batched=True,
    )

def tokenize(tokenizer: PreTrainedTokenizer, text: str, max_length: int, logger: logging.Logger):
    if tokenizer.padding_side == "right":
        logger.warning("Padding side is right, setting it to left")
        tokenizer.padding_side = "left"
    if max_length is None:
        padding = "longest"
    else:
        padding = "max_length"
    return tokenizer(
        text,
        return_tensors="pt",
        return_token_type_ids=False,
        truncation=True,
        padding=padding,
        max_length=max_length,
    )
