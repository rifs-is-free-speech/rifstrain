"""Evaluate
========
This module contains the functions to evaluate the performance of the
trained model.

This module contains only one function:

    - evaluate_model: Evaluate the performance of the trained model.

"""

import os
import torch
import pandas as pd

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

    results_path = os.path.join(output_path, f"results_{model_name}_{dataset_name}.csv")
    if os.path.exists(results_path):
        if not quiet:
            print(f"Results file '{results_path}' already exists")
            print("Skipping evaluation")
        return

    experiments_path = os.path.join(output_path, "results.csv")
    if not os.path.exists(experiments_path):
        with open(experiments_path, "w+") as f:
            f.write("model,dataset,metric,value\n")

    if verbose and not quiet:
        print(f"Loading '{dataset_name}' dataset and initializing '{model_name}' model")
    processor = Wav2Vec2Processor.from_pretrained(pretrained_path)

    trnsfrms = [
        ToTensor(),
        RemoveSpecialCharacters(),
        PrepareDataset(processor=processor),
    ]

    test_dataset = SpeechDataset(
        csv_file=csv_test_file,
        transform=transforms.Compose(trnsfrms),
        shuffle=False,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=SpeechCollator(processor=processor, padding=True),
    )

    model = Wav2Vec2ForCTC.from_pretrained(pretrained_path).to(device)
    metrics = compute_metrics(processor=processor)

    if verbose and not quiet:
        print("Starting evaluation")
    results = []
    for i, sample in enumerate(test_dataloader):
        if verbose and not quiet:
            print(f"Processing sample {i+1} of {len(test_dataloader)}")

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

        result = metrics.calculate_metrics(
            predictions=prediction, references=references
        )

        if verbose and not quiet:
            print(f"Prediction: {prediction[0]}")
            print(f"Reference: {references[0]}")
            print("\n".join([f"{k}: {v*100:.2f}%" for k, v in result.items()]) + "\n")

        result["prediction"] = prediction[0]
        result["reference"] = references[0]
        results.append(result)
    df = pd.DataFrame(results)

    wer = df["WER"].mean()
    cer = df["CER"].mean()
    levenshtein_ratio = df["LSR"].mean()

    if verbose and not quiet:
        print(
            f"Avg. wer: {wer}, Avg. cer: {cer}, Avg. levenshtein ratio: {levenshtein_ratio}"
        )

    with open(experiments_path, "a") as f:
        for metric in ["WER", "CER", "LSR"]:
            f.write(f"{model_name},{dataset_name},{metric},{df[metric].mean()}\n")

    df.to_csv(results_path, index=False, header=True)
