"""rifstrain
=========

This package contains the code for training and finetuning ASR models.
"""

from rifstrain.finetune import finetune
from rifstrain.evaluate import evaluate

__version__ = "0.0.1"

__all__ = ["finetune", "evaluate"]
