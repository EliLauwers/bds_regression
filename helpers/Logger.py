import datetime
import json
import numpy as np


class Logger:
    """
    Used to log to a log file or to a model evaluates file
    """

    instance = (
        None  # This instance is declared before the init to use as a plt type of class
    )

    def __init__(self, log_path, model_evaluates_path):
        Logger.instance = self
        self.log_path = log_path
        self.first_print = True

        # Init a new json path
        self.model_evaluates_path = model_evaluates_path
        with open(self.model_evaluates_path, "w") as file:
            file.write(json.dumps([]))

    def create_logger(log_path, model_evaluates_path):
        """
        Used to make sure there can only be one active logger
        """
        if Logger.instance:
            return Logger.instance
        return Logger(log_path, model_evaluates_path)

    def process(self, text_of_interest=None, add_to_file=True):
        """
        Used to log to the console and a general log file
        :param text_of_interest: String to be displayed in the log file
        :return: None
        """
        # First, print a 'new logger initiated' when the logger has been called for the first time
        if self.first_print:
            with open(self.log_path, "a") as file:
                timestamp = str(datetime.datetime.now().date())
                file.write("\n")
                file.write(f"------ NEW LOGGER {timestamp} ".ljust(75, "-"))
                file.write("\n")
            self.first_print = False

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        text = f"{timestamp} - {text_of_interest}\n"

        if add_to_file:
            with open(self.log_path, "a") as file:
                file.write(text)
        print(text[:-1])

    def evaluate_predictions(self, meta, predictions, y_true):
        predictions_exp = np.exp(predictions)
        y_true_exp = np.exp(y_true)

        preds_centered = predictions_exp - y_true_exp
        bias = np.mean(preds_centered)
        standard_error = np.std(preds_centered)
        R2 = np.square(np.corrcoef(predictions_exp, y_true_exp))[0, 1]

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        eval_dict = {
            "timestamp": timestamp,
            "meta": meta,
            "metrics": {"bias": bias, "standard_error": standard_error, "R2": R2},
        }
        print(eval_dict)

        # first read, then write the data
        with open(self.model_evaluates_path, "r+") as file:
            cur_file = json.load(file)

        with open(self.model_evaluates_path, "w") as file:
            cur_file.append(eval_dict)
            file.write(json.dumps(cur_file))

        model_name, agg_name = eval_dict["meta"].values()
        bias, R2, std = eval_dict["metrics"].values()
        print(bias, R2, std)
        self.process(
            f"Model fit done!\n{'-' * 8} - {model_name} with {agg_name}\n{'-' * 8} - Bias: {round(bias, 2)}, R2: {round(R2,2)}, Std: {round(std,2)}"
        )
        
        
