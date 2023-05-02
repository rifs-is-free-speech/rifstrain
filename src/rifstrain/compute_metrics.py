"""Compute metrics
===============

Compute the different metric used to evaluate.

Right now we only implement the following metrics:

    - WER

This is by default what is loaded when you use compute_metrics
"""

from evaluate import load
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
        self.processor = processor

    def __call__(self, pred):
        """Compute the WER metric.

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
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
