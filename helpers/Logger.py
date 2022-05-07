import datetime
import json
class Logger:
    def __init__(self, log_path, model_evaluates_path):
        self.log_path = log_path
        with open(self.log_path, 'a') as file:
            timestamp = str(datetime.datetime.now().date())
            file.write('\n')
            file.write(f"------ NEW LOGGER {timestamp} ".ljust(75,'-'))
            file.write('\n')
        self.model_evaluates_path = model_evaluates_path
        # Init a new json path
        with open(self.model_evaluates_path, 'w') as file:
            file.write(json.dumps([]))

    def process(self, text_of_interest = None):
        """
        Used to log to the console and a general log file
        :param text_of_interest: String to be displayed in the log file
        :return: None
        """
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        text = f"{timestamp} - {text_of_interest}\n"
        with open(self.log_path, 'a') as file:
            file.write(text)
        print(text[:-1])

    def model_evaluates(self, eval_dict):
        """
        Writes away a models evaluative parameters to a general log file
        :param evals: a dictionary containing all relevant information
        :return: None
        """
        if type(eval_dict) != dict:
            raise AssertionError("model evals should be a dictionary")
        if sorted(eval_dict.keys()) != ["meta", "metrics"]:
            raise AssertionError("Please only use keys 'meta' and 'metrics'")
        eval_dict['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # first read, then write the data
        with open(self.model_evaluates_path, 'r+') as file:
            cur_file = json.load(file)

        with open(self.model_evaluates_path, 'w') as file:
            cur_file.append(eval_dict)
            file.write(json.dumps(cur_file))

