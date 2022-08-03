#!/usr/bin/env python

import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()
from datasets import Dataset

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lightning_transformers.dataio import DataCollatorWithPadding, DefaultDataCollator, DynamicBatchSampler

# logging.basicConfig(level=logging.INFO)


set_verbosity_error()


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def count_samples(dataloader):
    true_samples = 0
    tot_samples = 0
    n_batches = 0
    t1 = time.time()
    for batch in dataloader:
        # audio, lens = batch.signal
        # true_samples += torch.sum(audio.shape[-1] * lens).item()
        # padded_samples += torch.sum(audio.shape[-1] * (1 - lens)).item()
        # print(audio.shape)

        # print(batch)
        # n = batch["input_ids"].shape[0]
        # print(n)

        tot_samples += batch["attention_mask"].numel()
        true_samples += batch["attention_mask"].sum().item()
        n_batches += 1

    elapsed = time.time() - t1
    # tot_samples = true_samples + padded_samples

    ratio_true = true_samples / tot_samples

    return ratio_true, 1 - ratio_true, n_batches, elapsed


def main():
    # csv_path = "/home/ubuntu/bhuang/nlu/nlu-data/axa/train.csv"
    # csv_path = "/home/ubuntu/bhuang/nlu/nlu-data/lfm/v1/train.csv"
    csv_path = "/home/ubuntu/bhuang/nlu/nlu-data/ag2r/train.csv"

    df = pd.read_csv(csv_path)
    df.columns = ["input", "label"]
    sentences = df["input"].tolist()

    raw_datasets = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    outputs = tokenizer(sentences, padding=False, truncation=False, return_attention_mask=False, return_token_type_ids=False)
    input_ids = outputs["input_ids"]
    lengths_list = np.asarray([len(inp) for inp in input_ids])
    max_seq_len = int(
        np.percentile(
            lengths_list, 97, interpolation="nearest"
        )
    )
    print(f"max_seq_len has been set to {max_seq_len}")

    def preprocess_function(examples, padding, truncation, max_length):
        sentences = examples["input"]
        # todo: return_type, att_mask
        # ! pad format
        result = tokenizer(sentences, padding=padding, truncation=truncation, max_length=max_length, return_token_type_ids=False)

        return result

    processed_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, "max_length", True, max_seq_len),
        batched=True,
        remove_columns=raw_datasets.column_names,
        desc="Running tokenizer on dataset",
    )

    # processed_datasets = TensorDataset(processed_datasets)

    default_data_collator = DefaultDataCollator()

    # ?
    data_collator_w_padding = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if False else None))

    batch_size = 32
    max_batch_len = max_seq_len * 32
    num_buckets = 10

    # random
    dataloader = DataLoader(processed_datasets, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
    ratio_true, ratio_padding, n_batches, elapsed = count_samples(dataloader)
    print("Random Sampling: ratio of true samples {:.4f}, ratio of padding {:.4f}, #batches {}, Total time {:.4f}s".format(ratio_true, ratio_padding, n_batches, elapsed))

    # sort
    # sorted_data = processed_datasets.filtered_sorted(sort_key="length")
    # dataloader = DataLoader(sorted_data, collate_fn=data_collator, batch_size=batch_size)
    # percent_true, percent_padded, n_batches, elapsed = count_samples(dataloader)
    # print("After sorting: % True samples {}, % of padding {}, Total time {}".format(percent_true, percent_padded, elapsed))

    processed_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, False, True, None),
        batched=True,
        remove_columns=raw_datasets.column_names,
        desc="Running tokenizer on dataset",
    )
    # ? eos
    # lengths_list = np.asarray([len(inp) for inp in processed_datasets["input_ids"]])

    t1 = time.time()
    dynamic_batcher = DynamicBatchSampler(
        processed_datasets,
        max_batch_length=max_batch_len,
        num_buckets=num_buckets,
        # lengths_list=lengths_list,
        length_func=lambda x: len(x["input_ids"]),
        shuffle=False,
        batch_ordering="descending",
        mode="lognorm",
        fit_dataset=False,
    )
    elapsed_sampler = time.time() - t1
    dataloader = DataLoader(processed_datasets, batch_sampler=dynamic_batcher, collate_fn=data_collator_w_padding)
    ratio_true, ratio_padding, n_batches, elapsed = count_samples(dataloader)
    print("Dynamic Batching w/ fixed lognorm: ratio of true samples {:.4f}, ratio of padding {:.4f}, #batches {}, Total time {:.4f}s, Sampler initialization time {:.4f}s".format(ratio_true, ratio_padding, n_batches, elapsed, elapsed_sampler))

    t1 = time.time()
    dynamic_batcher = DynamicBatchSampler(
        processed_datasets,
        max_batch_length=max_batch_len,
        num_buckets=num_buckets,
        # lengths_list=lengths_list,
        length_func=lambda x: len(x["input_ids"]),
        shuffle=False,
        batch_ordering="descending",
        mode="lognorm",
        fit_dataset=True,
    )
    elapsed_sampler = time.time() - t1
    dataloader = DataLoader(processed_datasets, batch_sampler=dynamic_batcher, collate_fn=data_collator_w_padding)
    ratio_true, ratio_padding, n_batches, elapsed = count_samples(dataloader)
    print("Dynamic Batching w/ fit lognorm: ratio of true samples {:.4f}, ratio of padding {:.4f}, #batches {}, Total time {:.4f}s, Sampler initialization time {:.4f}s".format(ratio_true, ratio_padding, n_batches, elapsed, elapsed_sampler))

    t1 = time.time()
    dynamic_batcher = DynamicBatchSampler(
        processed_datasets,
        max_batch_length=max_batch_len,
        num_buckets=num_buckets,
        # lengths_list=lengths_list,
        length_func=lambda x: len(x["input_ids"]),
        shuffle=False,
        batch_ordering="descending",
        mode="kmeans",
    )
    elapsed_sampler = time.time() - t1
    dataloader = DataLoader(processed_datasets, batch_sampler=dynamic_batcher, collate_fn=data_collator_w_padding)
    ratio_true, ratio_padding, n_batches, elapsed = count_samples(dataloader)
    print("Dynamic Batching w/ Kmeans: ratio of true samples {:.4f}, ratio of padding {:.4f}, #batches {}, Total time {:.4f}s, Sampler initialization time {:.4f}s".format(ratio_true, ratio_padding, n_batches, elapsed, elapsed_sampler))


if __name__ == "__main__":
    main()
