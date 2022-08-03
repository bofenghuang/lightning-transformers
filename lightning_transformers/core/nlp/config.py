from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List

from lightning_transformers.core.config import TaskConfig
from lightning_transformers.core.data import TransformerDataConfig


@dataclass
class HFTransformerDataConfig(TransformerDataConfig):
    # load dataset from HF hub
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    # load dataset from local files
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    # train test split
    train_val_split: Optional[int] = None
    limit_train_samples: Optional[int] = None
    limit_val_samples: Optional[int] = None
    limit_test_samples: Optional[int] = None
    # tokenizer
    padding: Union[str, bool] = "max_length"
    truncation: Union[str, bool] = "only_first"
    max_length: Optional[int] = 128
    max_length_pctl: Optional[float] = None
    # hf datasets map
    preprocessing_batch_size: Optional[int] = 1000
    preprocessing_num_workers: Optional[int] = None
    load_from_cache_file: bool = True
    cache_dir: Optional[Union[Path, str]] = None


@dataclass
class HFTokenizerConfig:
    # ? useless
    downstream_model_type: Optional[str] = None
    pretrained_model_name_or_path: Optional[str] = None
    use_fast: bool = True


@dataclass
class HFBackboneConfig:
    pretrained_model_name_or_path: Optional[str] = None


@dataclass
class HFTaskConfig(TaskConfig):
    downstream_model_type: Optional[str] = None
    backbone: HFBackboneConfig = HFBackboneConfig()