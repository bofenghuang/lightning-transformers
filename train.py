"""The shell entry point `$ pl-transformers-train` is also available."""
import hydra
from omegaconf import DictConfig


import logging

# logging.basicConfig(
#     format="%(asctime)s [%(levelname)s] [nlp-lit] [%(filename)s:%(lineno)d] %(message)s",
#     datefmt="%Y-%m-%dT%H:%M:%SZ",
# )

# logging_level = logging.INFO
# logging_fmt = "%(levelname)s:%(name)s:%(message)s"
# logging_fmt = "%(asctime)s [%(levelname)s] [nlp-lit] [%(filename)s:%(lineno)d] %(message)s"

log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [nlp-lit] [%(filename)s:%(lineno)d] %(message)s", "%Y-%m-%dT%H:%M:%SZ",
)

# console_handler = logging.StreamHandler()
# console_handler.setFormatter(log_formatter)
# logger.addHandler(console_handler)

# file_handler = logging.FileHandler("{0}/{1}.log".format(log_path, filename))
# file_handler.setFormatter(log_formatter)
# logger.addHandler(file_handler)


# root_logger = logging.getLogger()
# root_logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# if not root_logger.hasHandlers():
#     root_logger.addHandler(console_handler)
#     root_logger.propagate = False

# ISO_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
# BASIC_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(module)s] [%(funcName)s] %(message)s"
# logging.basicConfig(
#     format=BASIC_LOG_FORMAT, datefmt=ISO_DATE_FORMAT, level=logging.INFO
# )

from pytorch_lightning import _logger
_logger.handlers[0].setFormatter(log_formatter)

from transformers.utils import logging as hg_logging
# hg_logging.set_verbosity_info()
# leave it empy or use a string
hf_logger = hg_logging.get_logger()
hf_logger.handlers[0].setFormatter(log_formatter)

from datasets import logging as ds_logging
# ds_logging.set_verbosity_info()
# leave it empy or use a string
ds_logger = ds_logging.get_logger()
# ds_logger.handlers[0].setFormatter(log_formatter)
ds_logger.addHandler(console_handler)
ds_logger.propagate = False

# try:
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging_level)
#     root_handler = root_logger.handlers[0]
#     root_handler.setFormatter(logging.Formatter(logging_fmt))
# except IndexError:
#     logging.basicConfig(level=logging_level, format=logging_fmt)

from lightning_transformers.cli.train import main


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
