# Copyright 2022 Zaion Lab (authors: Bofeng Huang)
# coding=utf-8

# import itertools
import logging
from typing import Iterable, List, Optional, Tuple

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MultiLabelBinarizer

# todo
# logger = logging.getLogger(__name__)


def encode_labels(labels: List) -> Tuple[Iterable, List]:
    """Encode labels for multi class classification."""
    # classes = sorted_labels(labels)
    # # convert labels to label ids
    # label_ids = labels_to_label_ids(classes, labels)

    le = LabelEncoder()
    return le.fit_transform(labels), le.classes_.tolist()


def split_multi_labels(labels: List, sep: str = "+") -> List:
    """Split string multi labels, convert label into list."""
    return [lab.split(sep) for lab in labels]


def exclude_labels(labels: List, labels_to_exclude: List) -> List:
    """Remove irrelevant labels."""
    return [[lab for lab in label if lab not in labels_to_exclude] for label in labels]


def binarize_labels(labels: List) -> Tuple[Iterable, List]:
    """Binary encoding labels."""
    lb = LabelBinarizer()
    return lb.fit_transform(labels), lb.classes_.tolist()


def binarize_multi_labels(labels: List) -> Tuple[Iterable, List]:
    """Binary encoding multi labels."""
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(labels), mlb.classes_.tolist()


def convert_labels(
    labels: List,
    problem_type: str,
    multi_label_sep: Optional[str] = None,
    labels_to_exclude: Optional[List] = None,
) -> Optional[Tuple[Iterable, List]]:
    """Convert labels to label ids.

    For `single_label_classification`, encode labels to label ids.
    For `multi_label_classification`, binarize labels into onehot encoded lists.
    """
    if problem_type == "single_label_classification":
        # classes = sorted_labels(labels)
        # convert labels to label ids
        # label_ids = labels_to_label_ids(classes, labels)

        label_ids, classes = encode_labels(labels)
    elif problem_type == "multi_label_classification":
        labels = split_multi_labels(labels, sep=multi_label_sep)
        if labels_to_exclude:
            # this will force multi label cls model to learn that
            # examples of this class don't belong to any other label
            labels = exclude_labels(labels, labels_to_exclude)
        # labels to onehot encoded label ids
        label_ids, classes = binarize_multi_labels(labels)
    else:
        raise NotImplementedError(f'"{problem_type}" problem type is not supported yet.')

    return label_ids, classes


# obsolete
def sorted_labels(labels: List):
    """
    Sort labels.
    """
    return sorted(list(set(labels)))
    # return sorted(list(set(itertools.chain.from_iterable(labels))))


# obsolete
def labels_to_label_ids(classes: List, labels: List):
    """
    Convert labels to label ids.
    """
    # TODO : exception
    label_ids = [classes.index(lab) for lab in labels]
    return label_ids
