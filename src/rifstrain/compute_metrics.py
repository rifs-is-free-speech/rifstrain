"""Compute metrics
===============

Compute the different metric used to evaluate.

Right now we implement the following metrics:

    - WER
    - CER
    - Levenshtein Ratio

This is by default what is loaded when you use compute_metrics
"""

from evaluate import load
from Levenshtein import ratio
import numpy as np


class compute_metrics:
    """
    Compute metrics for the model.
    """

    def __init__(self, processor):
        """Initialize the class.

        Parameters
        ----------
        processor: Wav2Vec2Processor
            Processor to use.
        """
        self.wer_metric = load("wer")
        self.cer_metric = load("cer")
        self.processor = processor

    def calculate_metrics(self, predictions: str, references: str):
        """Calculate the metrics.

        Parameters
        ----------
        predictions: str
            Prediction string.
        references: str
            Label string.

        Returns
        -------
        dict
            Dictionary containing the metrics.

        """

        if len(predictions) == 1 or len(references) == 1:
            LSR = ratio(predictions[0], references[0])
        else:
            avg_LSR = 0
            for i in range(len(predictions)):
                avg_LSR += ratio(predictions[i], references[i])
            LSR = avg_LSR / len(predictions)
        wer = self.wer_metric.compute(predictions=predictions, references=references)
        cer = self.cer_metric.compute(predictions=predictions, references=references)
        return {"WER": wer, "CER": cer, "LSR": LSR}

    def __call__(self, pred):
        """Compute the metrics.

        Parameters
        ----------
        pred: EvalPrediction
            Prediction to compute the metric on.

        Returns
        -------
        dict
            Dictionary containing the WER metric.

        """
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)
        return self.calculate_metrics(pred_str, label_str)
