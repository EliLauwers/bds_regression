from helpers.Logger import Logger
import pandas as pd
import pickle

RANDOM_STATE = 0
LOG = Logger.create_logger("logs/log_file.txt", "logs/model_evaluative_parameters.json")

# Used in Bootstrap
with open("data/intermediate/pre_processed/y_train.pk", "rb") as infile:
    y_true = pickle.load(infile)

BOOTSTRAP_OBS = len(y_true)
BOOTSTRAP_B = 10
