import pandas as pd
from helpers.validate_date_transformation import validate_date_transformation
from GLOBAL_VARS import LOG
import pickle

def create_dataset():
    LOG.process("CREATE DATASET")
    LOG.process("Reading data")
    music_data = pd.read_csv("data/raw/music_data.csv")
    metadata = pd.read_csv("data/raw/metadata.csv")
    meta = metadata[["track_id","track_listens","album_date_released","album_id"]]
    dates = pd.to_datetime(metadata["album_date_released"],format="%Y-%m-%d %H:%M:%S")
    validate_date_transformation(raw_dates = metadata["album_date_released"], transformed_dates=dates)
    metadata.drop("album_date_released",axis=1)
    metadata["album_date_released"] = dates
    LOG.process("Joining Data")
    joined = music_data.merge(meta, on ="track_id",how="inner")
    LOG.process("Writing Data")
    with open("data/intermediate/joined_data.pk", "wb") as file:
        pickle.dump(joined, file, protocol= pickle.HIGHEST_PROTOCOL)
    LOG.process("Dataset Created!\n")