"""
Datasets
========

Not to be confused with rifsdatasets (which is a package for reading and
writing datasets), this module contains the classes for the datasets for
training. The classes contained in the module are:

    - SpeechDataset
    - SpeechCollator

"""

import os
import torch

import numpy as np
import pandas as pd
import soundfile as sf

from typing import Dict, List, Union, Callable
from torch.utils.data import Dataset

from transformers import AutoProcessor


class SpeechDataset(Dataset):
    """SpeechDataset implements a generic Dataset class for the speech datasets"""

    def __init__(
        self,
        csv_file: str,
        transform: Callable = None,
        shuffle: bool = False,
        validation: bool = False,
    ):
        """

        Parameters
        ----------
        csv_file : str
            Path to the csv file containing the dataset.
        transform : Callable, optional
            Optional transform to be applied on a sample.
        shuffle : bool
            Whether to shuffle the dataset.
        validation : bool
            Whether to use the dataset for validation. And reduce its size.
        """

        utterances = pd.read_csv(csv_file, sep=",", header=0)
        self.columns = list(utterances.columns)
        self.utterances = utterances.to_numpy()

        if shuffle:
            np.random.shuffle(self.utterances)

        if validation:
            if len(self.utterances) > 1000:
                self.utterances = self.utterances[:1000]
            self.utterances = self.utterances[:1000]

        self.dataset_path = os.path.dirname(csv_file)
        self.transform = transform

    def __len__(self):
        """
        Return the length of the dataset.

        Returns
        -------
        length : int
            Length of the dataset.
        """
        return len(self.utterances)

    def __getitem__(self, index: int):
        """
        Return the item at the given index.

        Parameters
        ----------
        index : int
            Index of the item to return.

        Returns
        -------
        item : dict
            Dictionary containing the item.
        """
        item = self.utterances[index]

        target_txt = item[self.columns.index("text")]
        file_path = item[self.columns.index("id")]

        # TODO: Look in alignments directory
        audio_array, sampling_rate = sf.read(os.path.join(self.dataset_path, file_path))

        sample = {
            "target_txt": target_txt,
            "audio_array": audio_array,
            "file_path": file_path,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class SpeechCollator:
    """
    Data collator that will dynamically pad the inputs received.

    Parameters
    ----------
    processor:
        The processor used for proccessing the data.
    padding: Union[bool, str, PaddingStrategy]
        Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:
        * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
        sequence if provided).
        * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
        maximum acceptable input length for the model if that argument is not provided.
        * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
        different lengths).
    """

    def __init__(self, processor: AutoProcessor, padding: Union[bool, str] = True):
        """Constructor method for the SpeechCollator class."""
        self.processor = processor
        self.padding = padding

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Pad inputs (on left/right and up to predefined length or max length in the batch)

        Parameters
        ----------
        features : List[Dict[str, Union[List[int], torch.Tensor]]]
            List of dictionary of input values

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of padded values
        """
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch
