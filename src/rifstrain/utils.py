"""Utils
=====

Utility functions for the training.

This includes the following classes:

    - ToTensor
    - RemoveSpecialCharacters
    - PrepareDataset

"""

import re
import torch

chars_to_ignore_regex = '[\,\?\.\!\-\;\:"]'  # noqa: W605


class ToTensor:
    """Convert ndarray in sample to Tensors."""

    def __call__(self, sample):
        """Convert ndarray in sample to Tensor.

        Parameters
        ----------
        sample : dict
            Dictionary containing the sample.

        Returns
        -------
        sample : dict
            Dictionary containing the sample.
        """
        audio_array = sample["audio_array"]
        sample["audio_array"] = torch.from_numpy(audio_array)
        return sample


class RemoveSpecialCharacters:
    """Remove special characters from the transcript."""

    def __call__(self, sample):
        """Remove special characters from the transcript.

        Parameters
        ----------
        sample : dict
            Dictionary containing the sample.

        Returns
        -------
        sample : dict
            Dictionary containing the sample.
        """
        sample["target_txt"] = re.sub(
            chars_to_ignore_regex, "", sample["target_txt"]
        ).lower()

        return sample


class PrepareDataset:
    """Prepare the dataset for the model."""

    def __init__(self, processor):
        """Constructor for the PrepareDataset class.

        Parameters
        ----------
        processor : Wav2Vec2Processor
            Processor to use.
        """
        self.processor = processor

    def __call__(self, sample):
        """Prepare the dataset for the model.

        Parameters
        ----------
        sample : dict
            Dictionary containing the sample.

        Returns
        -------
        sample : dict
            Dictionary containing the sample.
        """
        sample["input_values"] = self.processor(
            sample["audio_array"], sampling_rate=16000
        ).input_values[0]
        sample["input_length"] = len(sample["input_values"])
        sample["labels"] = self.processor(text=sample["target_txt"]).input_ids

        return sample
