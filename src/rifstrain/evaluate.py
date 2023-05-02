"""
Evaluate file
"""

import os
import torch
import shutil

import evaluate as ev
from torchvision import transforms

from rifs.model.settings import ModelSettings

from rifs.datasets import SpeechDataset, SpeechCollator

from utils import (
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
    tsv_test_file: str,
    data_path: str,
    model_path: str,
    output_path: str,
    noise_pack: str,
    run_id: str,
    eval_snapshots: bool = True,
    max_predict: int = 100,
    move_files: bool = False,
):
    """
    Evaluates the model on the given dataset. Can also evaluate intermediate models saved at steps.

    Parameters
    ----------
    tsv_test_file : str
        Path to the test tsv file.
    data_path : str
        Path to the dataset folder.
    model_path : str
        Path to the model file.
    output_path : str
        Path to the output folder.
    noise_pack : str
        String denoting the noise pack.
    run_id : str
        ID of the run.
    eval_snapshots : bool
        Whether to evaluate intermediate models saved at steps.
    max_predict : int
        Maximum number of predictions to make.
    """

    # Load settings and wer metrics
    ms = ModelSettings()
    wer_metric = ev.load("wer")

    clean_data_path = os.path.join(
        data_path, f"clean_noise_data_{noise_pack}" if noise_pack else "clean_data"
    )

    # Create output folder
    output_path = os.path.join(output_path, run_id)
    os.makedirs(os.path.join(output_path, "audio"), exist_ok=True)

    # Load absolute paths for all models to evaluate
    models_to_test = [model_path]
    if eval_snapshots:
        snapshot_folder = os.path.join(model_path, "wave2vec2-base-da-snapshot")
        models_to_test += [
            os.path.join(snapshot_folder, m) for m in os.listdir(snapshot_folder)
        ]

    resultsfile = f"results_{noise_pack}.csv" if noise_pack else "results.csv"

    # Results file for all models
    with open(os.path.join(output_path, resultsfile), "w+") as f:
        f.write("model,wer\n")

    processor = Wav2Vec2Processor.from_pretrained(models_to_test[0])

    trnsfrms = [
        ToTensor(),
        RemoveSpecialCharacters(),
        PrepareDataset(processor=processor),
    ]

    test_dataset = SpeechDataset(
        tsv_test_file,
        clean_data_path,
        transforms.Compose(trnsfrms),
        shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=SpeechCollator(processor=processor, padding=True),
    )

    for model in models_to_test:
        model_name = os.path.basename(model)

        # Load model
        model = Wav2Vec2ForCTC.from_pretrained(model).to(device)

        resultsmodelfile = (
            f"results_{model_name}_{noise_pack}.csv"
            if noise_pack
            else f"results_{model_name}.csv"
        )
        if max_predict:
            with open(os.path.join(output_path, resultsmodelfile), "w+") as f:
                f.write("prediction,labels,wer,soundfile\n")

        # Initialise the WER metric
        wer = 0
        for i, sample in enumerate(test_dataloader):

            # Feed sample to model
            input_dict = processor(
                sample["input_values"],
                sampling_rate=ms.sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            logits = model(input_dict["input_values"].squeeze(1).to(device)).logits
            pred_ids = torch.argmax(logits, dim=-1)

            # Decode the predictions and get references
            prediction = processor.batch_decode(pred_ids)
            references = processor.batch_decode(sample["labels"])

            # Compute WER for this sample
            sample_wer = wer_metric.compute(
                predictions=prediction, references=references
            )
            wer += sample_wer

            if i < max_predict and max_predict:
                utterance_id = test_dataset[i]["utterance_id"]
                wavfilename = (
                    f"{utterance_id[5:9]}/{utterance_id[5:15]}/{utterance_id}.wav"
                )
                new_wavfilename = (
                    f"{i}_{utterance_id}_{noise_pack}.wav"
                    if noise_pack
                    else f"{i}_{utterance_id}.wav"
                )

                with open(os.path.join(output_path, resultsmodelfile), "a") as f:
                    f.write(
                        f"{prediction[0]},{references[0]},{sample_wer},{new_wavfilename}\n"
                    )

                if not move_files:
                    continue
                wavfile_src = os.path.join(clean_data_path, wavfilename)
                wavfile_dst = os.path.join(output_path, "audio", new_wavfilename)
                if not os.path.exists(wavfile_dst):
                    shutil.copyfile(wavfile_src, wavfile_dst)

        # Compute average WER
        wer = wer / len(test_dataloader)
        with open(os.path.join(output_path, resultsfile), "a") as f:
            f.write(f"{model_name},{wer}\n")
