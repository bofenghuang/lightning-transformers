from typing import Any

import hydra
import torch
from dalle_pytorch.dalle_pytorch import DiscreteVAE
from torchvision.utils import save_image

from lightning_transformers.core import TaskTransformer
from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator


class VQVAE(TaskTransformer):
    def __init__(
        self,
        instantiator: Instantiator,
        backbone: Any,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        **config_data_args,
    ):
        self.save_hyperparameters()
        model: DiscreteVAE = hydra.utils.instantiate(backbone, **config_data_args)
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        loss = self.model(images, return_recon_loss=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, "test")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, "validation")

    def common_step(self, batch: Any, prefix) -> torch.Tensor:
        import torch.nn.functional as F

        images, labels = batch
        loss = self.model(images, return_recon_loss=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        if prefix == "test":
            output = self.model(images)
            loss = F.mse_loss(images, output)
            print("LOSS", loss)
            print(images)
            print(output)
            save_image(images, "input.png")
            save_image(output, "output.png")
            raise ValueError
        return loss
