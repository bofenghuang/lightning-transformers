# Copyright 2022 Zaion Lab (authors: Bofeng Huang)
# coding=utf-8

import csv
import json
import logging
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from datasets import Dataset
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn

from lightning_transformers.task.nlp.text_classification.config import TextClassificationPreprocessDataConfig
# from lightning_transformers.task.nlp.text_classification.preprocess_label import convert_labels

# todo: use pl logger
# logger = logging.getLogger(__name__)


class TextClassificationPreprocessDataModule:
    pattern_ponct = re.compile("[%s]" % re.escape(string.punctuation.replace("'", "")))

    def __init__(
        self,
        cfg: TextClassificationPreprocessDataConfig = TextClassificationPreprocessDataConfig(),
    ) -> None:
        self.cfg = cfg

    def preprocess(
        self,
        ds: Dataset,
    ) -> Optional[Dataset]:
        cfg = self.cfg

        if ds.get("train"):
            rank_zero_info('Preprocess the "training" dataset')
            ds["train"] = TextClassificationPreprocessDataModule.preprocess_data(
                ds["train"],
                input_field=cfg.input_field,
                label_field=cfg.label_field,
                do_norm_inputs=cfg.train_do_norm_inputs,
                do_norm_labels=cfg.train_do_norm_labels,
                do_clean_duplicates=cfg.train_do_clean_duplicates,
                min_examples_per_class=cfg.train_min_examples_per_class,
                preprocessing_batch_size=cfg.preprocessing_batch_size,
                preprocessing_num_workers=cfg.preprocessing_num_workers,
            )
        else:
            raise ValueError("Failed when preprocessing dataset, training set is not provided")

        if ds.get("validation"):
            rank_zero_info('Preprocess the "validation" dataset')
            ds["validation"] = TextClassificationPreprocessDataModule.preprocess_data(
                ds["validation"],
                input_field=cfg.input_field,
                label_field=cfg.label_field,
                do_norm_inputs=cfg.eval_do_norm_inputs,
                do_norm_labels=cfg.eval_do_norm_labels,
                do_clean_duplicates=cfg.eval_do_clean_duplicates,
                preprocessing_batch_size=cfg.preprocessing_batch_size,
                preprocessing_num_workers=cfg.preprocessing_num_workers,
            )

        if ds.get("test"):
            rank_zero_info('Preprocess the "test" dataset')
            ds["test"] = TextClassificationPreprocessDataModule.preprocess_data(
                ds["test"],
                input_field=cfg.input_field,
                label_field=cfg.label_field,
                do_norm_inputs=cfg.eval_do_norm_inputs,
                do_norm_labels=cfg.eval_do_norm_labels,
                do_clean_duplicates=cfg.eval_do_clean_duplicates,
                preprocessing_batch_size=cfg.preprocessing_batch_size,
                preprocessing_num_workers=cfg.preprocessing_num_workers,
            )

        # unify dataset attribute names for customized dataset
        if cfg.input_field != "input":
            ds = ds.rename_column(cfg.input_field, "input")
        if cfg.label_field != "label":
            ds = ds.rename_column(cfg.label_field, "label")

        return ds

    # def convert_labels(self, ds: Dataset) -> Tuple[Dataset, List]:
    #     """Encode or binarize labels in train/val/test dataset using HF datasets map function."""
    #     cfg = self.cfg

    #     ds_template = Dataset.from_dict({cfg.input_field: list(), cfg.label_field: list()})
    #     ds_train = ds.get("train", ds_template)
    #     ds_val = ds.get("validation", ds_template)
    #     ds_test = ds.get("test", ds_template)

    #     num_train = len(ds_train)
    #     num_val = len(ds_val)
    #     num_test = len(ds_test)

    #     all_labels_ids, classes = convert_labels(
    #         ds_train[cfg.label_field] + ds_val[cfg.label_field] + ds_test[cfg.label_field],
    #         problem_type=cfg.problem_type,
    #         multi_label_sep=cfg.multi_label_sep,
    #         labels_to_exclude=cfg.labels_to_exclude,
    #     )

    #     if num_train > 0:
    #         ds["train"] = ds_train.map(
    #             lambda _, idx: {cfg.label_field: all_labels_ids[:num_train][idx]},
    #             with_indices=True,
    #         )
    #     if num_val > 0:
    #         ds["validation"] = ds_val.map(
    #             lambda _, idx: {cfg.label_field: all_labels_ids[num_train : num_train + num_val][idx]},
    #             with_indices=True,
    #         )
    #     if num_test > 0:
    #         ds["test"] = ds_test.map(
    #             lambda _, idx: {
    #                 cfg.label_field: all_labels_ids[num_train + num_val : num_train + num_val + num_test][idx]
    #             },
    #             with_indices=True,
    #         )

    #     return ds, classes

    @staticmethod
    def parse_json(json_data: Dict) -> Optional[Tuple[List, List, Dict]]:
        corpus = json_data.get("intents", [])
        if not corpus:
            rank_zero_warn('Failed when parsing json, didn\'t find the "intents" attribute')
            return

        inputs = []
        labels = []
        for label_examples in corpus:
            # get intent name
            label = label_examples.get("intent", None)
            if label is None:
                # TODO : add warning
                continue
            # get examples of the intent
            examples = label_examples.get("examples", [])
            # check if has 'intent' attribute or if no null val
            # 'examples' ..
            if not examples:
                rank_zero_warn("Didn't find any example for class `%s`", label)
                continue

            for example in examples:
                sentence = example.get("text", "")
                if sentence:
                    inputs.append(sentence)
                    labels.append(label)

        # empty data
        if not inputs or not labels:
            rank_zero_warn("Failed when parsing json, didn't get any example")
            return

        # training hyper parameters
        hyper_params = json_data.get("hyperparameters", {})

        return inputs, labels, hyper_params

    @staticmethod
    def load_from_csv(csv_path: str) -> Optional[Tuple[List, List]]:
        inputs, labels = [], []
        with open(csv_path) as f:
            reader = csv.reader(f, delimiter=",")
            # skip the headers
            next(reader, None)

            for line in reader:
                inputs.append(",".join(line[:-1]))
                labels.append(line[-1])

        if not inputs or not labels:
            return

        return inputs, labels

    @staticmethod
    def norm_input(text: str) -> str:
        # convert to lowercase
        text = text.lower()
        # del ponctuations
        text = TextClassificationPreprocessDataModule.pattern_ponct.sub(" ", text)
        # del extra space
        text = " ".join(text.split())

        return text

    @staticmethod
    def norm_label(text: str) -> str:
        # del espace
        text = text.strip()

        return text

    @staticmethod
    def norm_data(
        ds: Dataset,
        input_field: str = "input",
        label_field: str = "label",
        do_norm_inputs: bool = False,
        do_norm_labels: bool = False,
        preprocessing_batch_size: Optional[int] = 1000,
        preprocessing_num_workers: Optional[int] = None,
    ) -> Tuple[Any, int]:
        """Normalize the input sentences and the labels using the HF datasets map function."""

        def norm_(
            example_batch: Any,
            input_field: str = "input",
            label_field: str = "label",
        ) -> str:
            if do_norm_inputs:
                example_batch[input_field] = [
                    TextClassificationPreprocessDataModule.norm_input(inp) for inp in example_batch[input_field]
                ]
            if do_norm_labels:
                example_batch[label_field] = [
                    TextClassificationPreprocessDataModule.norm_label(lab) for lab in example_batch[label_field]
                ]

            # if do_norm_inputs:
            #     example_batch[input_field] = TextClassificationPreprocessDataModule.norm_input(
            #         example_batch[input_field]
            #     )
            # if do_norm_labels:
            #     example_batch[label_field] = TextClassificationPreprocessDataModule.norm_label(
            #         example_batch[label_field]
            #     )
            return example_batch

        new_ds = ds.map(
            norm_,
            batched=True,
            batch_size=preprocessing_batch_size,
            num_proc=preprocessing_num_workers,
            fn_kwargs={"input_field": input_field, "label_field": label_field},
        )

        new_ds = new_ds.filter(lambda example: example[input_field] and example[label_field])

        return new_ds, len(ds) - len(new_ds)

    @staticmethod
    def get_dup_ids(arr: Iterable) -> Set:
        """Get indices of duplicate examples."""
        seen = set()
        # set add() method always returns None
        dup_ids = {i for i, x in enumerate(arr) if x in seen or seen.add(x)}
        return dup_ids

    @staticmethod
    def del_dup_examples(
        ds: Dataset,
        input_field: str = "input",
        label_field: str = "label",
        allow_duplicates_across_classes: bool = False,
    ) -> Tuple[Any, int]:
        """Remove duplicate examples."""
        if allow_duplicates_across_classes:
            dup_ids = TextClassificationPreprocessDataModule.get_dup_ids(zip(ds[input_field], ds[label_field]))
        else:
            dup_ids = TextClassificationPreprocessDataModule.get_dup_ids(ds[input_field])

        selected_ids = {i for i in range(len(ds)) if i not in dup_ids}
        new_ds = ds.select(selected_ids)

        return new_ds, len(dup_ids)

    @staticmethod
    def del_minor_classes(
        ds: Dataset,
        label_field: str = "label",
        min_examples_per_class: int = 5,
    ) -> Tuple[Any, int]:
        """Remove the minor classes without sufficient examples."""
        # count class examples num
        class_counts = defaultdict(int)
        for lab in ds[label_field]:
            class_counts[lab] += 1

        # remove minor classes
        del_classes = set()
        for cls, cnt in class_counts.items():
            # ignore class name with specific symbol
            # if safe_symbol_in_label is not None and safe_symbol_in_label in cls:
            #     continue
            if cnt < min_examples_per_class:
                del_classes.add(cls)
                rank_zero_warn(
                    "Expected %s examples for the class %s but got %s, this class will be deleted.",
                    min_examples_per_class,
                    cls,
                    cnt,
                )

        new_ds = ds.filter(lambda example: example[label_field] not in del_classes)

        return new_ds, len(ds) - len(new_ds)

    @staticmethod
    def preprocess_data(
        ds: Dataset,
        input_field: str = "input",
        label_field: str = "label",
        do_norm_inputs: bool = False,
        do_norm_labels: bool = False,
        do_clean_duplicates: bool = False,
        min_examples_per_class: Optional[int] = None,
        preprocessing_batch_size: Optional[int] = 1000,
        preprocessing_num_workers: Optional[int] = None,
    ) -> Optional[Dataset]:
        """Preprocess dataset including normalization, removing duplicates and removing minor classes"""
        if do_norm_inputs or do_norm_labels:
            ds, num_invalid = TextClassificationPreprocessDataModule.norm_data(
                ds,
                input_field=input_field,
                label_field=label_field,
                do_norm_inputs=do_norm_inputs,
                do_norm_labels=do_norm_labels,
                preprocessing_batch_size=preprocessing_batch_size,
                preprocessing_num_workers=preprocessing_num_workers,
            )
            if len(ds) == 0:
                rank_zero_warn("Failed when processing data, got 0 example after the data cleaning")
                return

        if do_clean_duplicates:
            # allow duplicate examples between different classes
            # for multi label classification
            # but not for multi class classification
            # unused, multi label training data should have string labels, separated by "+"
            ds, num_dup = TextClassificationPreprocessDataModule.del_dup_examples(
                ds,
                input_field=input_field,
                label_field=label_field,
                # allow_duplicates_across_classes=allow_duplicates_across_classes,
            )

        if min_examples_per_class is not None:
            ds, num_del = TextClassificationPreprocessDataModule.del_minor_classes(
                ds,
                input_field=input_field,
                label_field=label_field,
                min_examples_per_class=min_examples_per_class,
            )

        # todo
        num_labels = len(set(ds[label_field]))
        if num_labels < 2:
            rank_zero_warn("Failed when preprocessing data, please provide at least 2 classes")
            return

        rank_zero_info("Dataset has been preprocessed successfully")
        rank_zero_info("%s new examples added", len(ds))
        rank_zero_info("%s new intents added", num_labels)
        if do_clean_duplicates:
            rank_zero_info("%s duplicate examples ignored", num_dup)
        if do_norm_inputs or do_norm_labels:
            rank_zero_info("%s invalid examples ignored", num_invalid)
        if min_examples_per_class is not None:
            rank_zero_info("%s minor classes ignored", num_del)

        # class distribution
        class_unq, class_cnt = np.unique(ds[label_field], return_counts=True)
        rank_zero_info(
            "Num of examples by class: %s",
            json.dumps(
                dict(zip(class_unq, class_cnt.tolist())),
                indent=2,
                ensure_ascii=False,
            ),
        )

        return ds
