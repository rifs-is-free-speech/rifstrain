"""Fine tune
=========
Module for fine-tuning the ASR models

This module contains only one function:

    - finetune: fine-tune the ASR models

"""
import os
import sys
import torch
import warnings
import transformers

from rifstrain.compute_metrics import compute_metrics
from rifstrain.callbacks import CsvLogger, Timekeeper
from rifstrain.settings import ModelSettings, TrainerSettings
from rifstrain.datasets import SpeechDataset, SpeechCollator
from rifstrain.utils import (
    ToTensor,
    RemoveSpecialCharacters,
    PrepareDataset,
)

from rifstrain.vocab import write_vocab

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    TrainingArguments,
    Trainer,
)

from transformers.trainer_utils import get_last_checkpoint

from torchvision import transforms

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


def finetune(
    csv_train_file: str,
    csv_test_file: str,
    pretrained_path: str,
    hours: int = 60,
    minutes: int = 0,
    reduced_training_arguments: bool = True,
    model_save_location: str = "model",
    warmup_steps: int = 0,
    verbose: bool = False,
    quiet: bool = False,
    seed: int = 0,
):
    """
    Fine-tunes the model on the given dataset.

    Parameters
    ----------
    csv_train_file : str
        Path to the train csv file.
    csv_test_file : str
        Path to the test csv file.
    pretrained_path : str
        Path to the pretrained model.
    hours : int
        Number of hours to train for.
    minutes : int
        Number of minutes to train for in addition to hours.
    reduced_training_arguments : bool
        Whether to run a test run with reduced parameters.
    model_save_location : str
        Path to the directory where the model should be saved.
    warmup_steps : int
        Number of warmup steps.
    verbose : bool
        Whether to print the training progress.
    quiet : bool
        Whether to print nothing.
    seed : int
        Random seed.
    """
    dataset_path = os.path.dirname(csv_train_file)
    dataset_name = os.path.basename(dataset_path)
    model_name = os.path.basename(model_save_location)

    torch.manual_seed(seed)

    if verbose and not quiet:
        print(f"Dataset: {dataset_name}")
        print(f"Model: {model_name}")

    os.makedirs(model_save_location, exist_ok=True)
    os.makedirs(os.path.join(model_save_location, "checkpoints"), exist_ok=True)

    ms = ModelSettings()
    ts = TrainerSettings()

    if verbose and not quiet:
        print(f"Model settings: {ms}")
        print(f"Trainer settings: {ts}")

    write_vocab(dataset_path)
    tokenizer = Wav2Vec2CTCTokenizer(
        os.path.join(dataset_path, "vocab.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=ms.sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    if verbose and not quiet:
        print(f"tokenizer: {tokenizer}")
        print(f"feature_extractor: {feature_extractor}")
        print(f"processor: {processor}")

    trnsfrms_base = [
        ToTensor(),
        RemoveSpecialCharacters(),
        PrepareDataset(processor=processor),
    ]

    trnsfrms_train = transforms.Compose(trnsfrms_base)
    trnsfrms_dev = transforms.Compose(trnsfrms_base)

    if not quiet:
        print("Loading datasets")

    train_dataset = SpeechDataset(csv_train_file, trnsfrms_train, shuffle=True)
    test_dataset = SpeechDataset(
        csv_test_file, trnsfrms_dev, shuffle=True, validation=True
    )

    data_collator = SpeechCollator(processor=processor, padding=True)

    if verbose and not quiet:
        print("Loading model...")

    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_path,
        attention_dropout=ms.attention_dropout,
        hidden_dropout=ms.hidden_dropout,
        feat_proj_dropout=ms.feat_proj_dropout,
        mask_time_prob=ms.mask_time_prob,
        layerdrop=ms.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    ).to(device)

    model.freeze_feature_encoder()

    if verbose and not quiet:
        print(f"Model: {model}")

    if not quiet:
        print("Preparing training arguments")

    last_checkpoint = get_last_checkpoint(
        os.path.join(model_save_location, "checkpoints")
    )

    if not quiet and last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")

    training_args = TrainingArguments(
        output_dir=os.path.join(model_save_location, "checkpoints"),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        auto_find_batch_size=True,
        learning_rate=ts.lr,
        weight_decay=0.05,
        eval_accumulation_steps=1,
        num_train_epochs=sys.maxsize,
        group_by_length=True,
        evaluation_strategy="steps",
        dataloader_drop_last=False,
        logging_strategy="steps",
        logging_steps=10,
        eval_steps=1000,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=50000,
        gradient_checkpointing=True,
        warmup_steps=warmup_steps,
        save_total_limit=100,
        push_to_hub=False,
        log_level="info",
        resume_from_checkpoint=last_checkpoint,
        dataloader_num_workers=1,
    )

    if reduced_training_arguments:
        training_args = TrainingArguments(
            output_dir=os.path.join(model_save_location, "checkpoints"),
            auto_find_batch_size=True,
            learning_rate=ts.lr,
            weight_decay=0.05,
            gradient_accumulation_steps=1,
            eval_accumulation_steps=1,
            num_train_epochs=3,
            group_by_length=True,
            evaluation_strategy="steps",
            eval_steps=2,
            eval_delay=0,
            dataloader_drop_last=False,
            logging_strategy="steps",
            logging_steps=2,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=20,
            gradient_checkpointing=True,
            warmup_steps=3,
            save_total_limit=1,
            push_to_hub=False,
            resume_from_checkpoint=last_checkpoint,
        )

    if verbose and not quiet:
        print(f"Training arguments: {training_args}")

    if not quiet:
        print("Preparing trainer")

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics(processor=processor),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
        callbacks=[
            CsvLogger(
                save_location=os.path.join(model_save_location),
                model_name=model_name,
                dataset_name=dataset_name,
            ),
            Timekeeper(
                save_location=os.path.join(model_save_location),
                hours=hours,
                minutes=minutes,
            ),
        ],
    )

    if not quiet:
        print("Training")

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if not quiet:
        print("Saving model")

    trainer.save_model(model_save_location)
    processor.save_pretrained(model_save_location)
