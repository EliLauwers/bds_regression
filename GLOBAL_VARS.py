from helpers.Logger import Logger
import pickle
import os

RANDOM_STATE = 1234
LOG = Logger.create_logger("logs/log_file.txt", "logs/model_evaluative_parameters.json")

# Used in Bootstrap

if os.path.exists("data/intermediate/track_listens/y_train.pk"):
    with open("data/intermediate/track_listens/y_train.pk", "rb") as infile:
        y_train = pickle.load(infile)
        BOOTSTRAP_OBS = len(y_train)
# BOOTSTRAP_OBS = 100
BOOTSTRAP_B = 300
