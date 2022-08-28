from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from pydantic import parse_file_as
from matplotlib.patches import Rectangle

from hourglass_tensorflow.type import HTFPersonDatapoint

HTF_JSON = "data/htf.ignore.json"

CM_TAB20 = plt.cm.get_cmap("tab20")


def plot_histogram(
    x,
    bins,
    title="Histogram",
    xlabel="X",
    ylabel="Y",
    labels=list(),
    density=True,
    cm=CM_TAB20,
):
    fig, ax = plt.subplots()

    _, bins, patches = ax.hist(x=x, bins=bins, density=density)
    plt.title(label=title)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    # handles = []
    for c, p in enumerate(patches[::-1]):
        plt.setp(p, "facecolor", cm(c))
        # handles = [Rectangle((0, 0), 1, 1, color=cm(c), ec="k")] + handles
    ax.legend(handles=patches, labels=labels)
    plt.show()


def plot_stacked_bar(
    df: pd.DataFrame,
    title="",
    legend=True,
    xlabel="X",
    ylabel="Y",
):
    df.plot(
        kind="bar",
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        stacked=True,
    )
    if legend:
        plt.legend(loc="upper left", ncol=2)
    for n, x in enumerate([*df.index.values]):
        for (proportion, y_loc) in zip(df.loc[x], df.loc[x].cumsum()):
            plt.text(
                x=n - 0.17,
                y=(y_loc - proportion) + (proportion / 2),
                s=f"{np.round(proportion * 100, 1)}%",
                color="black",
                fontsize=12,
                fontweight="bold",
            )

    plt.show()


if __name__ == "__main__":
    # Parse file as list of records
    logger.info(f"Reading HTF data at {HTF_JSON}")
    data = parse_file_as(List[HTFPersonDatapoint], HTF_JSON)
    # Compute Stats
    ## Average number of joints and average number of visible joints
    num_joints = [len(d.joints) for d in data]
    num_visible_joints = [len([j for j in d.joints if j.visible]) for d in data]
    avg_joints_per_sample = np.mean(num_joints)
    avg_visible_joints_per_sample = np.mean(num_visible_joints)
    ## Joint ID distribution
    joints_id = [(j.id, j.visible) for d in data for j in d.joints]
    joints_df = pd.DataFrame(joints_id, columns=["Joint ID", "Visible"])
    joints_df.Visible = joints_df.Visible.astype(int).map(
        {0: "Not Visible", 1: "Visible"}
    )
    joints_crosstab = pd.crosstab(
        index=joints_df["Joint ID"], columns=joints_df["Visible"], normalize="index"
    )
    only_visible_joints_id = [jid for jid, j_visible in joints_id if j_visible]
    # Plots
    plot_histogram(
        x=num_joints,
        bins=16,
        title="Distribution of LABELED joints per sample",
        labels=[f"Joint {i}" for i in range(16)],
        xlabel="Number of labeled joints",
        ylabel="Density accross dataset",
    )
    plot_histogram(
        x=num_visible_joints,
        bins=16,
        title="Distribution of VISIBILE joints per sample",
        labels=[f"Joint {i}" for i in range(16)],
        xlabel="Number of visible joints",
        ylabel="Density accross dataset",
    )
    plot_stacked_bar(
        df=joints_crosstab,
        title="Visible distribution accross Joints",
        xlabel="Joint ID",
        ylabel="Proportion of visible joints",
        legend=True,
    )
