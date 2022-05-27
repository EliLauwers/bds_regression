from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import random
from GLOBAL_VARS import FWIDTH, FHEIGHT
from mpld3 import save_html

random.seed(0)
if __name__ == "__main__":

    ####
    #
    # Decision: We'll use the min max scaler
    #
    #####

    for variable in ["track_listens", "album_date_released"]:

        with open(f"data/intermediate/{variable}/X_train.pk", "rb") as file:
            data = pickle.load(file)

        # Check if we can do something by scaling
        k = 15
        cols = random.choices(data.columns, k=k)
        flierprops = dict(
            marker="o",
            markerfacecolor=(0.86, 0.86, 0.86, 0.2),
            markersize=2,
            markeredgecolor="grey",
        )
        fig, axes = plt.subplots(1, 3, figsize=(FWIDTH * 3, FHEIGHT), sharex=True)
        scalers = (StandardScaler, MinMaxScaler, RobustScaler)
        names = ["Standard", "MinMax", "Robust"]
        tmp = data[cols]
        for ax, scaler, name in zip(axes, scalers, names):
            data = scaler().fit(tmp).transform(tmp)
            ax.set_title(name)
            ax.boxplot(data, flierprops=flierprops)

        plt.suptitle(f"{k} Random variables scaled")
        output_path = f"plots/rs1_scaler/{variable}"
        plt.savefig(output_path + ".png", dpi=300)
        save_html(fig, output_path + ".html")
        plt.clf()
        pass
