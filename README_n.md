# LIT-NLP

- Configuration *(Hydra, omegaConf, dataclasses)*
  - Hierarchical parameterization
  - CLI, argument overriding
- DataModule *(pl.LightningDataModule, re, sklearn, datasets, tokenizers)*
  - Load datasets from *.csv, *.json, memory or public HF datasets
  - Split train, val and test
  - Preprocess (num_procs, bs)
    - Normalize text, check duplicates, delete minor classes
    - Tokenization, set "max_length"
    - Encode labels; Split, exclude, binarize labels (multi-label classification)
    - set_format
  - Dataloader
- Backbone *(torch.nn.Module, transformers)*
  - Define layers, forward
- System *(pl.LightningModule)
  - Losses (multi-label classification)
  - Optimizers, schedulers
  - Metrics (multi-label classification)
  - Train, eval logic
  - Postprocessing
    - Temperature scaling
    - Pruning, quantization
    - Export: onnx, tensorrt
Trainer *(pl.Trainer)*
  - Multi GPUs, multi nodes
  - Mixed precision
  - Callbacks: logger, checkpointer, earlystopper

## Usage

```bash
HYDRA_FULL_ERROR=1 python ./train.py dataset=nlp/text_classification/axa backbone=nlp/lang/fr criterion=nlp/text_classification/focal +trainer/callbacks=early_stopping trainer.max_epochs=100 trainer/logger=wandb log=True
```