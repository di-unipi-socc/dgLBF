from glob import iglob
from os.path import join

import config as c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from matplotlib.font_manager import FontProperties

FLOWS_UNIQUE = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
FILES = list(iglob(join(c.RESULTS_DIR, "*.csv")))


def save_plot(name: str):
    plt.savefig(
        c.PLOT_PATH.format(name=name),
        dpi=c.PLOT_DPI,
        format=c.PLOT_FORMAT,
        bbox_inches="tight",
    )

    print(f"✅ {name} plot")


def remove_outliers_iqr(df, field, threshold=1.5):
    # remove outliers using IQR (interquartile range)
    q1 = df[field].quantile(0.25)
    q3 = df[field].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)

    df = df[(df[field] > lower_bound) & (df[field] < upper_bound)]
    return df


# remove outlier with scipy.stats zscore for specified column
def remove_outliers_zscore(df, field, threshold=3):
    z = np.abs(zscore(df[field]))
    df.loc[z > threshold, field] = df[field].median()
    return df


def get_95_percentile(df, field):
    q = df[field].quantile(0.95)
    df = df[df[field] < q]
    return df


def remove_incomplete(df: pd.DataFrame):
    # remove Infrs that are not present in all Flows and runs
    counts = df["Infr"].value_counts()
    bound = len(FLOWS_UNIQUE) * len(FILES)
    df = df[df["Infr"].isin(counts[counts == bound].index)]

    return df


def clean_data(df: pd.DataFrame):

    # remove rows with NaN Time(s)
    df = df.dropna(subset=["Time"])

    # remove Infrs that are not present in all Flows and runs
    # df = remove_incomplete(df)

    # remove outliers
    df = remove_outliers_iqr(df, "Time")
    return df


def flows_time(df: pd.DataFrame, suffix: str = ""):
    # for each "Flows" unique value make the average of only the column "Time"

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Flows", y="Time", label="Time")

    plt.legend(loc="upper left")
    plt.ylabel("Time (s)")
    plt.xlim(0, FLOWS_UNIQUE[-1])

    save_plot(f"time-{suffix}")


def infr_flow_time(df: pd.DataFrame, suffix: str = ""):

    plt.figure(figsize=(10, 6))
    df1 = df.sort_values(by=["Nodes"])
    df1["Infr"] = df1["Infr"].str.replace("infr", "")
    sns.lineplot(
        data=df1,
        x="Flows",
        y="Time",
        # label=infr,
        # errorbar=None,
        hue=df1["Infr"] + " (" + df1["Nodes"].astype(str) + ")",
        style=df1["Infr"] + " (" + df1["Nodes"].astype(str) + ")",
        markers=True,
        dashes=False,
        palette="colorblind",
    )

    legend = plt.legend(title="infrastructure (#nodes)\n", ncol=1)
    plt.gca().add_artist(legend)

    font = FontProperties()
    font.set_weight("bold")
    legend.get_title().set_font_properties(font)

    plt.ylabel("Time (s)")
    plt.xlim(0, FLOWS_UNIQUE[-1])

    save_plot(f"infr-flow-time-{suffix}")


def flow_infr_time(df: pd.DataFrame, suffix: str = ""):

    plt.figure(figsize=(10, 6))
    df1 = df.sort_values(by=["Nodes"])
    df1["Infr"] = df1["Infr"].str.replace("infr", "")
    sns.lineplot(
        data=df1,
        x="Nodes",
        y="Time",
        hue="Flows",
        palette="colorblind",
    )

    legend = plt.legend(title="infrastructure (#nodes)\n", ncol=1)
    plt.gca().add_artist(legend)

    font = FontProperties()
    font.set_weight("bold")
    legend.get_title().set_font_properties(font)

    plt.ylabel("Time (s)")
    plt.xlim(0, FLOWS_UNIQUE[-1])

    save_plot(f"flow-infr-time-{suffix}")


if __name__ == "__main__":
    # Load data
    df = pd.concat(map(pd.read_csv, FILES), ignore_index=True)
    # df = pd.read_csv(join(c.RESULTS_DIR, "results-281194.csv"))

    # Clean data
    df = clean_data(df)

    # get the rows of Infr<number> infrastructures
    df_random = df[df["Infr"].str.contains(r"infr\d+")]
    df_topologies = df.drop(df_random.index)

    # Plot
    sns.set_theme(style="darkgrid")

    if not df.empty:
        pass
        # df["Time"] = df["Time"] * 1000  # convert to ms
        flows_time(df_random, suffix="random")
        infr_flow_time(df_random, suffix="random")
        flows_time(df_topologies, suffix="topologies")
        infr_flow_time(df_topologies, suffix="topologies")
    else:
        print("No data to plot!")
