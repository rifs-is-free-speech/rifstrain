"""Callbacks
=========

File containing callback functions used during Training.

We implement the following Callback options:

    - CsvLogger: Save training metrics to CSV file.
    - Timekeeper: Stop model training when time is up.

"""

import pendulum
import json

from os.path import join
from transformers import TrainerCallback


def dumper(obj: dict):
    """JSON serializer for objects not serializable by default json code

    Parameters
    ----------
    obj : object
        Object to be serialized.
    """
    try:
        return obj.toJSON()
    except AttributeError:
        return str(obj)


class CsvLogger(TrainerCallback):
    """
    Callback for logging training and evaluation metrics to a csv file.
    """

    def __init__(self, save_location: str, model_name: str, dataset_name: str):
        """Constructor for CsvLogger

        Parameters
        ----------
        save_location : str
            Location to save the CSV file.
        model_name : str
            Name of the model.
        dataset_name : str
            Name of the dataset.

        """
        self.save_location = save_location
        self.model_name = model_name
        self.dataset_name = dataset_name

    def on_init_end(self, args, state, control, **kwargs):
        """Callback function for logging training and evaluation metrics to a csv file.

        On trainer initialization will set up the csv file.

        Parameters
        ----------
        args
        state
        control
        kwargs
        """

        json_data = {
            "start_time": pendulum.now().to_iso8601_string(),
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            **vars(args),
            **vars(state),
        }

        with open(join(self.save_location, "run.json"), "w") as file:
            file.write(json.dumps(json_data, default=dumper, indent=4))

        with open(join(self.save_location, "metrics.csv"), "w+") as file:
            file.write("global_step,epoch,lr,loss,metrics\n")

    def on_evaluate(self, args, state, control, **kwargs):
        """Callback function for logging training and evaluation metrics to a csv file.

        On step on training step begin will write to CSV previous metrics if any

        Parameters
        ----------
        args
        state
        control
        kwargs
        """
        try:
            with open(join(self.save_location, "metrics.csv"), "a") as file:
                file.write(
                    f"{state.global_step},{state.epoch},{state.lr},{state.loss},{state.metrics}\n"
                )
        except:  # noqa
            pass


class Timekeeper(TrainerCallback):
    """
    Callback to stop training after a certain amount of time
    """

    def __init__(self, hours, minutes):
        """Constructor for Timekeeper

        Parameters
        ----------
        hours: int
            Number of hours to train for
        minutes: int
            Number of minutes to train for in addition to hours
        """
        self.last_print = pendulum.now().subtract(minutes=10)
        self.start_time = pendulum.now()
        self.end_time = self.start_time.add(hours=hours, minutes=minutes)

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
        if pendulum.now().subtract(minutes=10) > self.last_print:
            print(
                f"Time elapsed: {(pendulum.now() -self.start_time).in_words()}, Current step: {state.global_step}"
            )
            print(state)
            self.last_print = pendulum.now()
        if pendulum.now() > self.end_time:
            control.should_training_stop = True
            print("Stopping training")
