"""Evaluate
========
This module contains the functions to evaluate the performance of the
trained model.

This module contains only one function:

    - evaluate_model: Evaluate the performance of the trained model.

"""

import os
import torch

from torchvision import transforms
from rifstrain.compute_metrics import compute_metrics
from rifstrain.settings import ModelSettings
from rifstrain.datasets import SpeechDataset, SpeechCollator
from rifstrain.utils import (
    ToTensor,
    RemoveSpecialCharacters,
    PrepareDataset,
)
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(
    csv_test_file: str,
    data_path: str,
    pretrained_path: str,
    output_path: str,
    experiment_name: str,
    verbose: bool = False,
    quiet: bool = False,
):
    """
    Evaluates the model on the given dataset. Can also evaluate intermediate models saved at steps.

    Parameters
    ----------
    csv_test_file : str
        Path to the test csv file.
    data_path : str
        Path to the dataset folder.
    pretrained_path : str
        Path to the pretrained model.
    output_path : str
        Path to the output folder.
    experiment_name : str
        Name of the experiment.
    verbose : bool
        If True, prints the results of the evaluation progress.
    quiet : bool
        If True, does not print anything.
    """

    dataset_path = os.path.dirname(csv_test_file)
    dataset_name = os.path.basename(dataset_path)
    model_name = os.path.basename(pretrained_path)

    ms = ModelSettings()

    output_path = os.path.join(output_path, experiment_name)
    os.makedirs(os.path.join(output_path), exist_ok=True)

    resultsfile = f"results_{model_name}_{dataset_name}.csv"

    with open(os.path.join(output_path, resultsfile), "w+") as f:
        f.write("model,wer,cer,levenshtein_ratio\n")

    processor = Wav2Vec2Processor.from_pretrained(pretrained_path)

    trnsfrms = [
        ToTensor(),
        RemoveSpecialCharacters(),
        PrepareDataset(processor=processor),
    ]

    test_dataset = SpeechDataset(
        csv_test_file,
        dataset_path,
        transforms.Compose(trnsfrms),
        shuffle=False,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=SpeechCollator(processor=processor, padding=True),
    )

    model = Wav2Vec2ForCTC.from_pretrained(pretrained_path).to(device)
    metrics = compute_metrics(processor=processor)

    wer = 0
    for i, sample in enumerate(test_dataloader):

        input_dict = processor(
            sample["input_values"],
            sampling_rate=ms.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        logits = model(input_dict["input_values"].squeeze(1).to(device)).logits

        pred_ids = torch.argmax(logits, dim=-1)

        prediction = processor.batch_decode(pred_ids)
        references = processor.batch_decode(sample["labels"])

        results = metrics.calculate_metrics(
            predictions=prediction, references=references
        )
        if verbose and not quiet:
            print(results)
    with open(os.path.join(output_path, resultsfile), "a") as f:
        f.write(f"{model_name},{wer}\n")
