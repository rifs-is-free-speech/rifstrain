"""Callbacks
=========

File containing callback functions used during Training.

We implement the following Callback options:

    - CsvLogger
    - Timekeeper 

These are used to log training progress and make sure we 
gracefully stop models when they get close to the time limit.
"""

import pendulum
from transformers import TrainerCallback


class CsvLogger(TrainerCallback):
    """
    Callback for logging training and evaluation metrics to a csv file.
    """

    def __init__(self):
        """ Constructor for CsvLogger 

        Parameters
        ----------
        None

        """
        pass

    def on_init_end(self, args, state, control, **kwargs):
        json_data = {
            "run_id": self.run_id,
            "start_time": pendulum.now().to_iso8601_string(),
            "model": "Alvenir/Wav2VecForCTC",
            "lr": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
        }

        now = pendulum.now(tz="Europe/Paris").format("HH:mm:ss DD-MM-YYYY")
        
        json_data = {
            "run_id": self.run_id,
            "status": f"Init done - {now}",
        }

    def on_step_begin(self, args, state, control, **kwargs):
        json_data = {
            "run_id": self.run_id,
            "status": f"Running: Step {state.global_step} - {now}",
        }


class Timekeeper(TrainerCallback):
    """
    Callback to stop training after a certain amount of time
    """

    def __init__(self, hours, minutes, run_id, token):
        """
        Constructor for Timekeeper
        Parameters
        ----------
        hours: int
            Number of hours to train for
        minutes: int
            Number of minutes to train for in addition to hours
        run_id: str
            ID of the run
        """

        self.start_time = pendulum.now()
        self.end_time = self.start_time.add(hours=hours, minutes=minutes)
        self.run_id = run_id
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def on_step_end(self, args, state, control, **kwargs):
        """
        Stop training if time is up

        Parameters
        ----------
        args
        state
        control
        kwargs
        """

        if pendulum.now() > self.end_time:
            control.should_training_stop = True
            print("Stopping training")

            now = pendulum.now(tz="Europe/Paris").format("HH:mm:ss DD-MM-YYYY")
            json_data = {
                "run_id": self.run_id,
                "status": f"Done: Total steps {state.global_step} - {now}",
            }
            post_status(self.headers, json_data)
