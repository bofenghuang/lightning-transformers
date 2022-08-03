# Copyright 2022 Zaion Lab (authors: Bofeng Huang)

from dataclasses import dataclass
from typing import List, Optional, Union

from lightning_transformers.core.nlp import HFTransformerDataConfig


@dataclass
class TextClassificationPreprocessDataConfig:
    # do_norm_inputs: bool = True
    # do_norm_labels: bool = True
    # do_clean_duplicates: bool = False
    # allow_duplicates_across_classes: bool = True
    # min_examples_per_class: Optional[int] = None
    input_field: str = "input"
    label_field: str = "label"
    # train
    train_do_norm_inputs: bool = True
    train_do_norm_labels: bool = True
    train_do_clean_duplicates: bool = False
    train_min_examples_per_class: Optional[int] = None
    # val, test
    eval_do_norm_inputs: bool = False
    eval_do_norm_labels: bool = False
    eval_do_clean_duplicates: bool = False
    # label process
    # problem_type: str = "single_label_classification"
    # multi_label_sep: Optional[str] = None
    # labels_to_exclude: Optional[List] = None
    # process args
    preprocessing_batch_size: Optional[int] = 1000
    preprocessing_num_workers: Optional[int] = None


@dataclass
class TextClassificationDataConfig(HFTransformerDataConfig):
    # label
    problem_type: str = "single_label_classification"
    multi_label_sep: Optional[str] = None
    labels_to_exclude: Optional[List] = None


@dataclass
class TextClassificationTransformerConfig:
    downstream_model_type: str = "transformers.AutoModelForSequenceClassification"
    problem_type: str = "single_label_classification"


@dataclass
class TextClassificationCriterionConfig:
    type: Optional[str] = None
    # todo: other types
    weight: Optional[Union[int, List]] = None
    gamma: float = 0.2
    reduction: Optional[str] = None

    # todo: add ghm
