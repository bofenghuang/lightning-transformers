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
from typing import Any, Dict, Optional

import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall

from lightning_transformers.core.nlp import HFTransformer
from lightning_transformers.task.nlp.text_classification.config import (
    TextClassificationTransformerConfig,
    TextClassificationCriterionConfig,
)


class TextClassificationTransformer(HFTransformer):
    """Defines ``LightningModule`` for the Text Classification Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSequenceClassification``)
        **kwargs: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        cfg: TextClassificationTransformerConfig = TextClassificationTransformerConfig(),
        criterion: Optional[TextClassificationCriterionConfig] = None,
        **kwargs,
    ) -> None:
        self.cfg = cfg
        self.criterion_cfg = criterion
        downstream_model_type = cfg.get("downstream_model_type", "transformers.AutoModelForSequenceClassification")
        super().__init__(downstream_model_type, *args, **kwargs)
        self.metrics = {}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        cfg = self.cfg

        b_label_ids = batch.pop("labels")

        outputs = self.model(**batch)
        logits = outputs.logits

        if cfg.problem_type == "single_label_classification":
            loss = self.criterion(logits, b_label_ids)
        elif cfg.problem_type == "multi_label_classification":
            # loss = self.criterion(logits, b_label_ids.float())
            loss = self.criterion(logits, b_label_ids.type_as(logits))
        else:
            raise NotImplementedError(
                f'Failed running the training step, "{cfg.problem_type}" problem_type is not supported'
            )

        # outputs = self.model(**batch)
        # loss = outputs[0]

        self.log("train_loss", loss)
        # todo: metric logging
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        """Forward for evaluation step."""
        cfg = self.cfg

        b_label_ids = batch.pop("labels")

        outputs = self.model(**batch)
        logits = outputs.logits

        if cfg.problem_type == "single_label_classification":
            loss = self.criterion(logits, b_label_ids)
            # not needed for torchmetrics
            # preds = torch.argmax(logits, dim=1)
        elif cfg.problem_type == "multi_label_classification":
            # loss = self.criterion(logits, b_label_ids.float())
            loss = self.criterion(logits, b_label_ids.type_as(logits))
            # activ
            logits = logits.sigmoid()
        else:
            raise NotImplementedError(
                f'Failed running the {prefix} step, "{cfg.problem_type}" problem_type is not supported'
            )

        if b_label_ids is not None:
            metric_dict = self.compute_metrics(logits, b_label_ids, mode=prefix)
            self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)

        # outputs = self.model(**batch)
        # loss = outputs.loss
        # logits = outputs.logits
        # preds = torch.argmax(logits, dim=1)
        # if batch["labels"] is not None:
        #     metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
        #     self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        #     self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if -1 in batch["labels"]:
            batch["labels"] = None
        return self.common_step("test", batch)

    def configure_criterions(self):
        """Set objective function."""
        cfg = self.cfg
        criterion_cfg = self.criterion_cfg

        # todo: istancied by _target_ ?
        if criterion_cfg.type is None:
            if cfg.problem_type == "single_label_classification":
                from torch.nn import CrossEntropyLoss

                self.criterion = CrossEntropyLoss()
            elif cfg.problem_type == "multi_label_classification":
                from torch.nn import BCEWithLogitsLoss

                self.criterion = BCEWithLogitsLoss()
            else:
                raise NotImplementedError(
                    f'Failed when configuring the loss function, "{cfg.problem_type}" problem_type is not supported'
                )
        elif criterion_cfg.type == "focal":
            if cfg.problem_type == "single_label_classification":
                from lightning_transformers.loss import MultiClassFocalLoss

                # todo
                self.criterion = MultiClassFocalLoss(reduction=criterion_cfg.reduction)
            elif cfg.problem_type == "multi_label_classification":
                from lightning_transformers.loss import MultiLabelFocalLossWithLogits

                self.criterion = MultiLabelFocalLossWithLogits(
                    weight=None, gamma=criterion_cfg.gamma, reduction=criterion_cfg.reduction
                )
            else:
                raise NotImplementedError(
                    f'Failed when configuring the loss function, "{cfg.problem_type}" problem_type is not supported'
                )
        elif criterion_cfg.type == "ghm":
            if cfg.problem_type == "single_label_classification":
                from lightning_transformers.loss import MultiClassGHMCLoss

                # todo: device
                self.criterion = MultiClassGHMCLoss(reduction=criterion_cfg.reduction)
            elif cfg.problem_type == "multi_label_classification":
                from lightning_transformers.loss import MultiLabelGHMCLossWithLogits

                # todo: device
                self.criterion = MultiLabelGHMCLossWithLogits(reduction=criterion_cfg.reduction)
            else:
                raise NotImplementedError(
                    f'Failed when configuring the loss function, "{cfg.problem_type}" problem_type is not supported'
                )
        else:
            raise NotImplementedError(
                f'Failed when configuring the loss function, "{criterion_cfg.type}" criterion is not supported'
            )

    def setup(self, stage: str):
        """Override setup method."""
        super().setup(stage=stage)
        self.configure_criterions()

    def configure_metrics(self, _) -> None:
        # todo: macro -> micro
        self.prec = Precision(num_classes=self.num_classes)
        self.recall = Recall(num_classes=self.num_classes)
        self.f1 = F1Score(num_classes=self.num_classes)
        self.acc = Accuracy()
        self.metrics = {"precision": self.prec, "recall": self.recall, "f1-score": self.f1, "accuracy": self.acc}

    @property
    def num_classes(self) -> int:
        return self.trainer.datamodule.num_classes

    def compute_metrics(self, preds, labels, mode="val") -> Dict[str, torch.Tensor]:
        # Not required by all models. Only required for classification
        return {f"{mode}_{k}": metric(preds, labels) for k, metric in self.metrics.items()}

    @property
    def hf_pipeline_task(self) -> str:
        return "sentiment-analysis"
