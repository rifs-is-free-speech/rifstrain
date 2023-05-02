"""
Module for fine-tuning the Alvenir/wav2vec2-base-da model on the ftspeech dataset.
"""
import os
import sys
import torch
import warnings
import transformers

from rifstrain.compute_metrics import compute_metrics
from rifstrain.callbacks import StatusUpdater, CsvLogger, Timekeeper
from rifstrain.settings import ModelSettings, TrainerSettings
from rifstrain.datasets import SpeechDataset, SpeechCollator
from rifstrain.utils import (
    ToTensor,
    RemoveSpecialCharacters,
    PrepareDataset,
)

from rifstrain.vocab import write_vocab

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)

from torchvision import transforms

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


def finetune(
    tsv_train_file: str,
    tsv_test_file: str,
    data_path: str,
    model_path: str,
    from_pretrained_name: str,
    noise_pack: str,
    run_id: str,
    hours: int = 60,
    minutes: int = 0,
    test: bool = False,
    warmup_steps: int = 0,
):
    """
    Fine-tunes the model on the given dataset.

    Parameters
    ----------
    tsv_train_file : str
        Path to the train tsv file.
    tsv_test_file : str
        Path to the test tsv file.
    data_path : str
        Path to the dataset folder.
    model_path : str
        Path to the model file.
    from_pretrained_name : str
        Name of the pretrained model.
    noise_pack : str
        Name of the noise pack to use.
    run_id : str
        ID of the run.
    hours : int
        Number of hours to train for.
    minutes : int
        Number of minutes to train for in addition to hours.
    test : bool
        Whether to run a test run with reduced parameters.
    """

    ms = ModelSettings()
    ts = TrainerSettings()

    clean_data_path = os.path.join(
        data_path, f"clean_noise_data_{noise_pack}" if noise_pack else "clean_data"
    )

    write_vocab(clean_data_path)
    tokenizer = Wav2Vec2CTCTokenizer(
        os.path.join(clean_data_path, "vocab.json"),
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

    trnsfrms_base = [
        ToTensor(),
        RemoveSpecialCharacters(),
        PrepareDataset(processor=processor),
    ]

    trnsfrms_train, trnsfrms_dev = transforms.Compose(
        trnsfrms_base
    ), transforms.Compose(trnsfrms_base)

    # Load datasets
    train_dataset = SpeechDataset(tsv_train_file, clean_data_path, trnsfrms_train)
    test_dataset = SpeechDataset(tsv_test_file, clean_data_path, trnsfrms_dev)

    data_collator = SpeechCollator(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        from_pretrained_name,
        attention_dropout=ms.attention_dropout,
        hidden_dropout=ms.hidden_dropout,
        feat_proj_dropout=ms.feat_proj_dropout,
        mask_time_prob=ms.mask_time_prob,
        layerdrop=ms.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    ).to(device)

    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=os.path.join(model_path, "wave2vec2-base-da-snapshot"),
        auto_find_batch_size=True,
        learning_rate=ts.lr,
        weight_decay=0.05,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        num_train_epochs=sys.maxsize,
        group_by_length=True,
        evaluation_strategy="steps",
        eval_steps=100,
        eval_delay=0,
        dataloader_drop_last=False,
        logging_strategy="steps",
        logging_steps=100,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=10000,
        gradient_checkpointing=True,
        warmup_steps=warmup_steps,
        save_total_limit=15,
        push_to_hub=False,
    )
    if test:
        training_args = TrainingArguments(
            output_dir=os.path.join(model_path, "wave2vec2-base-da-snapshot"),
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
        )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics(processor=processor),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
        callbacks=[
            CsvLogger(),
            StatusUpdater(),
            Timekeeper(
                hours=hours,
                minutes=minutes,
                run_id=run_id,
            ),
        ],
    )

    trainer.train()

    trainer.save_model(model_path)
    processor.save_pretrained(model_path)
