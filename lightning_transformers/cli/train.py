# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_info

from lightning_transformers.core import TaskTransformer, TransformerDataModule
from lightning_transformers.core.config import TaskConfig, TrainerConfig, TransformerDataConfig
from lightning_transformers.core.instantiator import HydraInstantiator, Instantiator
from lightning_transformers.core.nlp.config import HFTokenizerConfig
from lightning_transformers.core.utils import set_ignore_warnings

import time


def run(
    instantiator: Instantiator,
    ignore_warnings: bool = True,
    run_test_after_fit: bool = True,
    dataset: TransformerDataConfig = TransformerDataConfig(),
    task: TaskConfig = TaskConfig(),
    trainer: TrainerConfig = TrainerConfig(),
    tokenizer: Optional[HFTokenizerConfig] = None,
    preprocessor: Optional[Any] = None,
    logger: Optional[Any] = None,
) -> None:
    if ignore_warnings:
        set_ignore_warnings()

    start_ = time.time()

    data_module_kwargs = {}
    if preprocessor is not None:
        data_module_kwargs["preprocessor"] = instantiator.instantiate(preprocessor)
        rank_zero_info("Preprocessor has been loaded")
    if tokenizer is not None:
        # data_module_kwargs["tokenizer"] = tokenizer
        data_module_kwargs["tokenizer"] = instantiator.instantiate(tokenizer)
        rank_zero_info("Tokenizer has been loaded")
    # data_module: TransformerDataModule = instantiator.data_module(dataset, **data_module_kwargs)
    data_module: TransformerDataModule = instantiator.instantiate(dataset, **data_module_kwargs)
    if data_module is None:
        raise ValueError("No dataset found. Hydra hint: did you set `dataset=...`?")
    if not isinstance(data_module, LightningDataModule):
        raise ValueError(
            "The instantiator did not return a DataModule instance." " Hydra hint: is `dataset._target_` defined?`"
        )
    # data_module attribute will be used to initialize model
    # ? can we save some setup execution
    data_module.setup(stage="fit")

    rank_zero_info(f"EVAL preprocessing: {time.time() - start_:.2f}s")
    start_ = time.time()

    model: TaskTransformer = instantiator.model(task, model_data_kwargs=getattr(data_module, "model_data_kwargs", None))

    trainer = instantiator.trainer(trainer, logger=logger,)

    rank_zero_info(f"EVAL loading: {time.time() - start_:.2f}s")
    start_ = time.time()

    trainer.fit(model, datamodule=data_module)

    rank_zero_info(f"EVAL training: {time.time() - start_:.2f}s")

    if run_test_after_fit:
        trainer.test(model, datamodule=data_module)


def main(cfg: DictConfig) -> None:
    # todo: rank_zero_info vs logging
    rank_zero_info(OmegaConf.to_yaml(cfg))
    instantiator = HydraInstantiator()
    logger = instantiator.logger(cfg)

    run(
        instantiator,
        ignore_warnings=cfg.get("ignore_warnings"),
        run_test_after_fit=cfg.get("training").get("run_test_after_fit"),
        dataset=cfg.get("dataset"),
        tokenizer=cfg.get("tokenizer"),
        preprocessor=cfg.get("preprocessor"),
        task=cfg.get("task"),
        trainer=cfg.get("trainer"),
        logger=logger,
    )


@hydra.main(config_path="../../conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
