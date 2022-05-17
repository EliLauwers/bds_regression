from turtle import title
from matplotlib import pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import random

random.seed(0)
if __name__ == "__main__":

    ####
    #
    # Decision: We'll use the min max scaler
    #
    #####

    with open("data/intermediate/joined_data.pk", "rb") as file:
        joined_data = pickle.load(file)

    joined_data = joined_data.drop(
        ["track_listens", "album_date_released"], axis=1
    ).set_index("track_id")

    # Check if we can do something by scaling
    cols = random.choices(joined_data.columns, k=20)
    tmp = joined_data[cols]
    tmp[:] = StandardScaler().fit(tmp).transform(tmp)
    tmp.columns = [str(20 - i) for i in range(tmp.shape[1])]
    tmp.plot(kind="box", vert=False, title="Random set of scaled variables")
    plt.savefig("plots/rs1_scaler/random_set_scaled_variables.png")
    plt.clf()
    # Note: Scaling will not work
    tmp = joined_data
    tmp[:] = MinMaxScaler().fit(joined_data).transform(joined_data)
    tmp = tmp[cols]
    tmp.columns = [str(20 - i) for i in range(tmp.shape[1])]
    tmp.plot(kind="box", vert=False, title="Random set of minmax variables")
    plt.savefig("plots/rs1_scaler/random_set_minmax_variables.png")
    plt.clf()
    # Note: try Robust Scaling
    tmp = joined_data
    tmp[:] = RobustScaler().fit(joined_data).transform(joined_data)
    tmp = tmp[cols]
    tmp.columns = [str(20 - i) for i in range(tmp.shape[1])]
    tmp.plot(kind="box", vert=False, title="Random set of robust variables")
    plt.savefig("plots/rs1_scaler/random_set_robust_variables.png")
    plt.clf()

    pass
