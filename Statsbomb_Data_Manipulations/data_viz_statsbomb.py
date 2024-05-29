
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mplsoccer import Pitch

def average_pass_location(events):  # todo if gonna use this, completely change the format of the catter - it looks shit
    """ Plots average pass and pass received location for a player.

        Parameters: events (pd.Dataframe): The required event data that is taken from Statsbomb.
     """
    passes = events[events["type_name"] == "Pass"]
    scatter_df = pd.DataFrame()
    for i, name in enumerate(passes["player_name"].unique()):
        passx = passes.loc[passes["player_name"] == name]["x"].to_numpy()
    recx = passes.loc[passes["pass_recipient_name"] == name]["end_x"].to_numpy()
    passy = passes.loc[passes["player_name"] == name]["y"].to_numpy()
    recy = passes.loc[passes["pass_recipient_name"] == name]["end_y"].to_numpy()
    scatter_df.at[i, "player_name"] = name
    scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
    scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))

    # Drawing pitch
    pitch = Pitch(line_color='grey')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                         endnote_height=0.04, title_space=0, endnote_space=0)
    # Scatter the location on the pitch
    pitch.scatter(scatter_df.x, scatter_df.y, s=500, color='yellow', edgecolors='green', linewidth=1,
                  alpha=1, ax=ax["pitch"], zorder=3)
    # annotating player name
    for i, row in scatter_df.iterrows():
        pitch.annotate(row.player_name, xy=(row.x, row.y), c='black', va='top', ha='center', weight="bold",
                       size=10, ax=ax["pitch"], zorder=4)

    fig.suptitle("Naismith average pass and reception location", fontsize=30)
    plt.show()

