# Copyright 2021 Zaion Lab (authors: Bofeng Huang)

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import ClassLabel, Dataset, Sequence, Value
from pytorch_lightning.utilities import rank_zero_info
from transformers import PreTrainedTokenizerBase

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.task.nlp.text_classification.config import TextClassificationDataConfig
from lightning_transformers.task.nlp.text_classification.preprocess_input import TextClassificationPreprocessDataModule
from lightning_transformers.task.nlp.text_classification.preprocess_label import (
    binarize_multi_labels,
    exclude_labels,
    split_multi_labels,
)

# todo: use pl logger
# logger = logging.getLogger(__name__)


class TextClassificationDataModule(HFDataModule):
    """Defines the ``LightningDataModule`` for Text Classification Datasets."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        preprocessor: Optional[TextClassificationPreprocessDataModule] = None,
        cfg: TextClassificationDataConfig = TextClassificationDataConfig(),
    ) -> None:
        super().__init__(tokenizer, cfg)
        self.preprocessor = preprocessor

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        """Process dataset including:
        - preprocessing (text normalization, duplicate examples checking, label encoding)
        - tokenization
        """
        cfg = self.cfg

        # preprocess
        if self.preprocessor is not None:
            dataset = self.preprocessor.preprocess(dataset)
            # dataset, self.classes = self.preprocessor.convert_labels(dataset)

        input_feature_fields = [k for k, v in dataset["train"].features.items() if k not in ["label", "idx"]]

        # set max_length if not setted
        if cfg.max_length is None and cfg.max_length_pctl is not None:
            self.cfg.max_length = TextClassificationDataModule.set_max_length(
                dataset["train"][input_feature_fields[0]], self.tokenizer, cfg.max_length_pctl
            )

        dataset = TextClassificationDataModule.preprocess(
            dataset,
            preprocessing_batch_size=cfg.preprocessing_batch_size,
            preprocessing_num_workers=cfg.preprocessing_num_workers,
            tokenizer=self.tokenizer,
            input_feature_fields=input_feature_fields,
            padding=cfg.padding,
            truncation=cfg.truncation,
            max_length=cfg.max_length,
        )

        cols_to_keep = [
            x
            for x in [
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "labels",
            ]
            if x in dataset["train"].features
        ]

        # todo : if public dataset
        # label to label ids
        if cfg.problem_type == "single_label_classification":
            if not isinstance(dataset["train"].features["labels"], ClassLabel):
                dataset = dataset.class_encode_column("labels")
                dataset.set_format("torch", columns=cols_to_keep)
                self.labels = dataset["train"].features["labels"]
        elif cfg.problem_type == "multi_label_classification":
            if not isinstance(dataset["train"].features["labels"], Sequence):
                dataset, classes = TextClassificationDataModule.class_binarize_column(
                    dataset,
                    input_field=input_feature_fields[0],
                    label_field="labels",
                    multi_label_sep=cfg.multi_label_sep,
                    labels_to_exclude=cfg.labels_to_exclude,
                )

                new_features = dataset["train"].features.copy()
                # ClassLabel force label to be int, not good for multi-label cls
                new_features["labels"] = Sequence(ClassLabel(names=classes))
                # new_features["labels"] = Sequence(Value("float"))
                dataset = dataset.cast(new_features)

                dataset.set_format("torch", columns=cols_to_keep)
                self.labels = dataset["train"].features["labels"].feature
                # fake a ClassLabel
                # self.labels = ClassLabel(names=classes)
        else:
            raise NotImplementedError(
                f'Failed when processing dataset, "{cfg.problem_type}" problem_type is not supported'
            )

        print(dataset["train"][0])

        return dataset

    @property
    def classes(self) -> List:
        return self.labels.names

    @property
    def num_classes(self) -> int:
        return self.labels.num_classes

    @property
    def model_data_kwargs(self) -> Dict[str, int]:
        return {"num_labels": self.num_classes}

    @staticmethod
    def convert_to_features(
        example_batch: Any, _, tokenizer: PreTrainedTokenizerBase, input_feature_fields: List[str], **tokenizer_kwargs
    ):
        # Either encode single sentence or sentence pairs
        if len(input_feature_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[input_feature_fields[0]],
                    example_batch[input_feature_fields[1]],
                )
            )
        else:
            texts_or_text_pairs = example_batch[input_feature_fields[0]]
        # Tokenize the text/text pairs
        return tokenizer(texts_or_text_pairs, **tokenizer_kwargs)

    @staticmethod
    def preprocess(
        ds: Dataset, preprocessing_batch_size: Optional[int], preprocessing_num_workers: Optional[int], **fn_kwargs
    ) -> Dataset:
        ds = ds.map(
            # todo: change this to self.convert_to_features for users to override
            TextClassificationDataModule.convert_to_features,
            batched=True,
            batch_size=preprocessing_batch_size,
            num_proc=preprocessing_num_workers,
            with_indices=True,
            fn_kwargs=fn_kwargs,
        )
        # todo
        # ds.rename_column_("label", "labels")
        ds = ds.rename_column("label", "labels")
        return ds

    @staticmethod
    def class_binarize_column(
        ds: Dataset,
        input_field: str = "input",
        label_field: str = "labels",
        multi_label_sep: Optional[str] = None,
        labels_to_exclude: Optional[List] = None,
    ) -> Tuple[Dataset, List]:
        ds_template = Dataset.from_dict({input_field: list(), label_field: list()})
        ds_train = ds.get("train", ds_template)
        ds_val = ds.get("validation", ds_template)
        ds_test = ds.get("test", ds_template)

        num_train = len(ds_train)
        num_val = len(ds_val)
        num_test = len(ds_test)

        all_labels = split_multi_labels(
            ds_train[label_field] + ds_val[label_field] + ds_test[label_field], sep=multi_label_sep
        )
        if labels_to_exclude:
            # this will force multi label cls model to learn that
            # examples of this class don't belong to any other label
            all_labels = exclude_labels(all_labels, labels_to_exclude)
        # labels to onehot encoded label ids
        all_labels_ids, classes = binarize_multi_labels(all_labels)
        # float
        # all_labels_ids = all_labels_ids.astype(float)

        if num_train > 0:
            ds["train"] = ds_train.map(
                lambda _, idx: {label_field: all_labels_ids[:num_train][idx]},
                with_indices=True,
            )
        if num_val > 0:
            ds["validation"] = ds_val.map(
                lambda _, idx: {label_field: all_labels_ids[num_train : num_train + num_val][idx]},
                with_indices=True,
            )
        if num_test > 0:
            ds["test"] = ds_test.map(
                lambda _, idx: {label_field: all_labels_ids[num_train + num_val : num_train + num_val + num_test][idx]},
                with_indices=True,
            )

        return ds, classes

    @staticmethod
    def set_max_length(input_sentences: List[str], tokenizer: PreTrainedTokenizerBase, max_length_pctl: float = 97):
        """Set the `max_length` attribute for tokenization"""
        # rank_zero_info('Setting the model parameter "max_seq_len"')
        # tokenize inputs and get len
        # truncate by model max acceptable length
        input_lens = [len(enc_tokens) for enc_tokens in tokenizer(input_sentences, truncation="only_first").input_ids]
        # stat input len
        rank_zero_info("Average number of tokens: {:.1f}".format((np.mean(input_lens))))
        rank_zero_info("Max number of tokens: %s", np.max(input_lens))

        # set max sequence length of bert input
        max_lenth = int(np.percentile(input_lens, max_length_pctl, interpolation="nearest"))
        # doesn't really make a difference empirically
        # self.config.max_seq_len = 128
        rank_zero_info('"max_seq_len" has been set to %s', max_lenth)

        # num_long_inputs = sum(np.asarray(input_lens) > max_lenth)
        # rank_zero_info(
        #     "{} reviews with length > {} ({:.2f} % of total data)".format(
        #         num_long_inputs,
        #         max_lenth,
        #         100 * num_long_inputs / len(input_lens),
        #     )
        # )

        return max_lenth
