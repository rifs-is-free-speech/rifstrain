"""Settings
========
Settings used for training and evaluating the model.

These can be changed to suit the user's needs. We recommend using the default
settings for the first run.

You can pass the settings via ENV variables or by changing the values here.
"""

from pydantic import BaseSettings


class ModelSettings(BaseSettings):
    """Model settings. These are loaded from the .env config file. prefix=MS_

    Some of these are not used for all models.
    """

    sampling_rate: int = 16000
    attention_dropout: float = 0.05
    hidden_dropout: float = 0.05
    feat_proj_dropout: float = 0.05
    mask_time_prob: float = 0.05
    layerdrop: float = 0.05

    class Config:
        """Config"""

        env_prefix = "MS_"
        env_file = ".env"
        env_file_encoding = "utf-8"


class TrainerSettings(BaseSettings):
    """Trainer settings. These are loaded from the .env config file. prefix=TS_

    Some of these are not used for all models.
    """

    batch_size: int = 16
    lr: float = 5e-7
    token: str = ""

    class Config:
        """Config"""

        env_prefix = "TS_"
        env_file = ".env"
        env_file_encoding = "utf-8"
