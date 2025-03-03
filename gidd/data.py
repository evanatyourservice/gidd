import functools
import os

import numpy as np
import torch
from transformers import BatchEncoding
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_dataset(config, num_proc=32):
    test_size = int(config.data.test_size)
    n_proc = min(os.cpu_count(), num_proc)
    train_ds = load_dataset(
        config.data.dataset_name,
        config.data.dataset_subset,
        split=f"train[:-{test_size}]",
        trust_remote_code=config.data.trust_remote_code,
        num_proc=n_proc,
    )
    test_ds = load_dataset(
        config.data.dataset_name,
        config.data.dataset_subset,
        split=f"train[-{test_size}:]",
        trust_remote_code=config.data.trust_remote_code,
        num_proc=n_proc,
    )

    return train_ds, test_ds


def default_collator(config, tokenizer, examples, text_key="text"):
    examples = [x[text_key] for x in examples]
    return tokenizer(examples, padding="max_length", truncation=True, max_length=config.model.max_seq_len, return_tensors="pt")


def subsample_collator(config, tokenizer, examples, text_key="text"):
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    examples = [x[text_key] for x in examples]
    tokens = tokenizer(examples, truncation=False, return_tensors="np")
    max_length = config.model.max_seq_len
    input_ids = []
    attn_masks = []
    for i in range(len(examples)):
        toks = tokens["input_ids"][i]
        attn_mask = tokens["attention_mask"][i]
        if toks[0] != bos_token_id:
            toks = np.concatenate([[bos_token_id], toks])
            attn_mask = np.concatenate([[1], attn_mask])
        if toks[-1] != eos_token_id:
            toks = np.concatenate([toks, [eos_token_id]])
            attn_mask = np.concatenate([attn_mask, [1]])

        if len(toks) > max_length:
            overflow = len(toks) - max_length
            start_idx = np.random.randint(0, overflow + config.data.max_add_padding)
            toks = toks[start_idx : start_idx + max_length]
            attn_mask = attn_mask[start_idx : start_idx + max_length]
        if len(toks) < max_length:
            underflow = max_length - len(toks)
            toks = np.pad(toks, (0, underflow), mode="constant", constant_values=tokenizer.pad_token_id)
            attn_mask = np.pad(attn_mask, (0, underflow), mode="constant", constant_values=0)
        assert len(toks) == max_length
        assert len(attn_mask) == max_length
        input_ids.append(toks)
        attn_masks.append(attn_mask)
    input_ids = torch.from_numpy(np.array(input_ids)).to(torch.long)
    attn_masks = torch.from_numpy(np.array(attn_masks)).to(torch.long)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attn_masks}, tensor_type="pt", n_sequences=len(input_ids))


def _get_dataloader(config, ds, shuffle, drop_last, batch_size, collate_fn):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = DistributedSampler(ds, seed=config.training.seed, shuffle=shuffle)
        _shuffle = False
    else:
        sampler = None
        _shuffle = shuffle

    return DataLoader(
        ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=config.data.num_workers,
        shuffle=_shuffle,
        pin_memory=True,
        persistent_workers=True,
    )


def get_dataloaders(config, tokenizer, train_batch_size=None, eval_batch_size=None):
    if train_batch_size is None:
        train_batch_size = config.training.train_batch_size
    if eval_batch_size is None:
        eval_batch_size = config.training.eval_batch_size

    train_ds, test_ds = get_dataset(config)

    collate_fn = functools.partial(subsample_collator, config, tokenizer, text_key="text")

    train_dl = _get_dataloader(config, train_ds, shuffle=True, drop_last=True, batch_size=train_batch_size, collate_fn=collate_fn)
    test_dl = _get_dataloader(config, test_ds, shuffle=False, drop_last=False, batch_size=eval_batch_size, collate_fn=collate_fn)

    return train_dl, test_dl