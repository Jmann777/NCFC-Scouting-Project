import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import logo

from highlight_text import fig_text
from mplsoccer import PyPizza, add_image
from scipy import stats

### Data transformations commented out for speed. Current code loads in csv that was created by the commented code

# Data import
# Commented out for speed
#with open('../data_logos/all_shots.pkl', 'rb') as file:
#    all_shots = pickle.load(file)
#    all_shots = all_shots.rename(columns={'id': 'id_x'})

#with open('../data_logos/all_events.pkl', 'rb') as file:
#    all_events = pickle.load(file)
#    all_events = all_events.rename(columns={'id': 'id_x'})

# Retrieving xG results from the model and combining into a player dataset
#with open('../data_logos/headed_shots_output.pkl', 'rb') as file:
#    headers_output = pickle.load(file)

#with open('../data_logos/regular_shots_output.pkl', 'rb') as file:
#    regular_shots_output = pickle.load(file)

#player_values = pd.read_csv('../data_logos/player_value_matched')
#player_minutes = pd.read_csv("../data_logos/player_minutes.csv")
#player_data = player_values.merge(player_minutes[['player_name', 'minutes_played']], on='player_name')

#all_xg_output = pd.concat([headers_output, regular_shots_output], ignore_index=True)

# Ensure there is a common identifier in all_shots and all_xg_output, e.g., 'shot_id'
# Here, we assume that both dataframes have a 'shot_id' column for merging
#if 'id_x' in all_shots.columns and 'id_x' in all_xg_output.columns:
    # Merge all_xg_output into all_shots based on 'shot_id'
#    all_shots = all_shots.merge(all_xg_output[['id_x', 'our_xg']], on='id_x', how='left')
#else:
#    raise KeyError("The necessary identifier 'id_x' is missing from one of the dataframes.")
# Replace NaN values with 0 in a specific column
#all_shots['our_xg'] = all_shots['our_xg'].fillna(0)


#players_xg_reg = regular_shots_output.groupby(["player_name"])["our_xg"].sum().sort_values(ascending=False).reset_index()
#players_xg_reg = players_xg_reg.rename(columns={'our_xg': 'regular_shots_xg'})
#players_xg_head = headers_output.groupby(["player_name"])["our_xg"].sum().sort_values(ascending=False).reset_index()
#players_xg_head = players_xg_head.rename(columns={'our_xg': 'headed_xg'})
#players_xg = pd.merge(players_xg_head, players_xg_reg, on="player_name", how='outer')
#players_xg = players_xg.fillna(0)
#players_xg['xg_total'] = players_xg['regular_shots_xg'] + players_xg['headed_xg']

#player_xg_scouting = player_data.loc[
#   (player_data["player_market_value_euro"] >= 1) &
#    (player_data["player_market_value_euro"] <= 10000000) &
#   (player_data["player_age"] < 26) &
#   (player_data["player_position"].isin(["Left Midfield", "Left Winger"]))]
#player_xg_scouting = player_xg_scouting.drop_duplicates(subset=['player_name'])
#player_xg_scouting = player_xg_scouting[player_xg_scouting["minutes_played"] > 450]

# Merging xg data with player data
#players_xg = player_xg_scouting.merge(players_xg, left_on='matched_player_name', right_on='player_name', how='left')
#players_xg["xg_p90"] = players_xg["xg_total"] / players_xg["minutes_played"] * 90
# Looking at Brady
#Brady_xg = player_xg_scouting[player_xg_scouting['player_name'] == "Robert Brady"]
# Finding the top 10 players with the highest xG
#top_10_players = players_xg.nlargest(10, 'xg_total')

# Creating other player metric - ** Commented out for spped
# Function to merge and calculate per 90 metrics
#def merge_and_calculate(df, metric_df, metric_name, calc_name):
#   df = pd.merge(df, metric_df, left_on='matched_player_name', right_on='player_name', how='left', suffixes=('', '_y'))
#   df.drop(columns=['player_name'], inplace=True)
#   df[calc_name] = (df[metric_name] / df["minutes_played"]) * 90
#   return df

# Goals
#players_g = all_shots[all_shots['outcome_name'] == 'Goal'].copy()
#players_g = players_g[players_g['player_name'].isin(players_xg['matched_player_name'])]
#players_g = players_g.groupby('player_name').size().reset_index(name='total_goals')
#players_xg = merge_and_calculate(players_xg, players_g, 'total_goals', 'goals_p90')
#players_xg['difference'] = (players_xg["total_goals"] - players_xg["xg_total"])

# Shots
#players_s = all_shots[all_shots['player_name'].isin(players_xg['matched_player_name'])]
#players_s = players_s.groupby('player_name').size().reset_index(name='total_shots')
#players_xg = merge_and_calculate(players_xg, players_s, 'total_shots', 'shots_p90')

# Dribbles
#players_d = all_events[all_events['type_name'] == 'Dribble'].copy()
#players_d = players_d[players_d['player_name'].isin(players_xg['matched_player_name'])]
#players_d = players_d.groupby('player_name').size().reset_index(name='total_take_ons')
#players_xg = merge_and_calculate(players_xg, players_d, 'total_take_ons', 'take_ons_p90')

# Carries
#players_c = all_events[all_events['type_name'] == 'Carry'].copy()
#players_c = players_c[players_c['player_name'].isin(players_xg['matched_player_name'])]
#players_c = players_c.groupby('player_name').size().reset_index(name='total_carries')
#players_xg = merge_and_calculate(players_xg, players_c, 'total_carries', 'carries_p90')

# Crosses
#players_cr = all_events[all_events['type_name'] == 'Pass'].copy()
#players_cr = players_cr[players_cr["pass_cross"] == True]
#players_cr = players_cr[players_cr['player_name'].isin(players_xg['matched_player_name'])]
#players_cr = players_cr.groupby('player_name').size().reset_index(name='total_crosses_attempted')
#players_xg = merge_and_calculate(players_xg, players_cr, 'total_crosses_attempted', 'crosses_attempted_p90')

# Assists
#players_a = all_events[all_events['pass_goal_assist'] == True].copy()
#players_a = players_a[players_a['player_name'].isin(players_xg['matched_player_name'])]
#players_a = players_a.groupby('player_name').size().reset_index(name='total_assists')
#players_xg = merge_and_calculate(players_xg, players_a, 'total_assists', 'assists_p90')

# Key passes
#players_kp = all_events[all_events['pass_shot_assist'] == True].copy()
#players_kp = players_kp[players_kp['player_name'].isin(players_xg['matched_player_name'])]
#players_kp = players_kp.groupby('player_name').size().reset_index(name='total_key_passes')
#players_xg = merge_and_calculate(players_xg, players_kp, 'total_key_passes', 'key_passes_p90')

# xA - Expected Assists
#pass_shot = all_events.loc[
#      (all_events["type_name"].isin(["Pass", "Shot"])) &
#       ((all_events['pass_shot_assist'] == True) | (all_events["type_name"] == "Shot"))
#]
#pass_shot = pass_shot.merge(all_shots[['id_x', "our_xg"]], how='left', on='id_x')
#pass_shot = pass_shot.sort_values(by=["possession", "index"])
#num_possession = max(pass_shot["possession"].unique())

#for i in range(num_possession + 1):
#    possession_chain = pass_shot.loc[pass_shot["possession"] == i].sort_values(by="index")
#    if len(possession_chain) > 0:
#        for j in range(len(possession_chain) - 1):
#            current_event = possession_chain.iloc[j]
#            next_event = possession_chain.iloc[j + 1]
#            if current_event["type_name"] == "Pass" and next_event["type_name"] == "Shot":
#                pass_shot.loc[current_event.name, 'our_xg'] = next_event["our_xg"]

#pass_shot_wxg = pass_shot.loc[pass_shot['pass_shot_assist'] == True]
#pass_shot_wxg = pass_shot_wxg[pass_shot_wxg['player_name'].isin(players_xg['matched_player_name'])]
#pass_shot_wxg = pass_shot_wxg.groupby('player_name')['our_xg'].sum().reset_index(name='total_xA')
#players_xg = merge_and_calculate(players_xg, pass_shot_wxg, 'total_xA', 'xA_p90')
#players_xg.fillna(0, inplace=True)
#players_xg.to_csv('../data_logos/player_xg.csv')


# Opening dataframe CSV
players_xg = pd.read_csv('../data_logos/player_xg.csv')
players_xg['xG xA combined'] = players_xg['xg_total'] + players_xg['total_xA']
# Looking at Brady
Brady_xg = players_xg[players_xg['player_name_x'] == "Robbie Brady"]
# Finding the top 10 players with the highest xG
top_10_players_g = players_xg.nlargest(10, 'xg_total')
top_10_players_a = players_xg.nlargest(10, 'total_xA')
top_10_players_op = players_xg.nlargest(10, 'difference')
top_10_players_com = players_xg.nlargest(10, 'xG xA combined')

# Creating variables for scatter
xG = players_xg['xg_total']
xA = players_xg['total_xA']
combined = players_xg['xG xA combined']
goals = players_xg['total_goals']
xG_overperformance = players_xg['difference']
val = players_xg['player_market_value_euro']


def player_scatter(x: pd.Series, y: pd.Series, diff: pd.Series, top_10_players,
                   yticks, xticks, x_tick_labels,
                   cbar_ticks, cbar_label,
                   title, xtitle, ytitle,
                   atitle, ayaxis):
    """ Creates a scatter plot of player statistics.

    Parameters:
- x (pd.Series): Series containing xG per 90 values for each player at Euro 2020 (min of 90 mins played to qualify).
    - y (pd.Series): Series containing the number of actual goals scored per player at Euro 2020 (min of 90 mins played)
    - diff (pd.Series): Series containing the difference values for coloring the scatter plot.
    - top_10_players (pd.DataFrame): DataFrame containing information of the top 10 players.
    - yticks (list): List of y-tick values.
    - xticks (list): List of x-tick values.
    - x_tick_labels (list): List of x-tick labels.
    - cbar_ticks (list): List of ticks for the color bar.
    - cbar_label (str): Label for the color bar.
    - title (str): Title of the scatter plot.
    - xtitle (str): Label for the x-axis.
    - ytitle (str): Label for the y-axis.
    - atitle (str): Column name for annotating points.
    - ayaxis (str): Column name for y-axis values in annotations.
    """
    fig = plt.figure(figsize=(8, 6), facecolor='#bebfc4')
    ax = plt.subplot(111, facecolor="#bebfc4")
    scatter = ax.scatter(x, y, s=100, c=diff, cmap='YlGn',
                         edgecolors='black', linewidths=1, alpha=0.75)
    # Customisation
    plt.style.use('seaborn-v0_8')
    cbar = fig.colorbar(scatter, ax=ax, label=cbar_label)
    cbar.set_ticks(cbar_ticks)
    ax.set_yticks(np.arange(0, 5, 1))
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', alpha=0.7)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Add logo
    logo.watermark(ax, 8, 6)
    # Annotate the top 10 players on the scatterplot
    for i, player in top_10_players.iterrows():
        plt.annotate(player['player_name_x'], (player[ayaxis], player[atitle]),
                     textcoords="offset points", xytext=(4, 4), ha='left')

    # Annotate Brady
    #for i, brady in Brady_xg.iterrows():
        #plt.annotate(brady['matched_player_name'], (brady[ayaxis], brady[atitle]),
                     #textcoords="offset points", xytext=(4, 4), ha='left', fontweight='bold')
    fig_text(
        x=0.09, y=0.97,
        s=title,
        fontname='Arial',
        color="black",
        fontweight='bold',
        size=20
    )

    fig_text(
        x=0.09, y=0.92,
        s="Viz by Josh Mann",
        fontname='Arial',
        color="#565756",
        size=14
    )

    plt.show()

# Prepping scatter function use for xG scatter
yticks = np.arange(0, 12, 1)
xticks = np.arange(0, max(val) + 1000000, 1000000)
x_tick_labels = [str(int(tick / 1000000)) for tick in xticks]
cbar_ticks = np.arange(0, 10.5, 1)
cbar_label = 'xG'
# Scatter for total xg
player_scatter(val, xG, xG, top_10_players_g,
               yticks, xticks, x_tick_labels,
               cbar_ticks, cbar_label,
               title="Top 5 leagues: xG vs Value", xtitle='Transfer value (€M)', ytitle='Expected Goals (xG)',
               atitle='xg_total', ayaxis='player_market_value_euro')


# Prepping scatter function use for xA scatter
yticks = np.arange(0, 12, 1)
xticks = np.arange(0, max(val) + 1000000, 1000000)
x_tick_labels = [str(int(tick / 1000000)) for tick in xticks]
cbar_ticks = np.arange(0, 11.5, 1)
cbar_label = 'xA'
# Scatter for xA vs value
player_scatter(val, xA, xA, top_10_players_a,
               yticks, xticks, x_tick_labels,
               cbar_ticks, cbar_label,
               title="Top 5 leagues: xA vs Value", xtitle='Transfer value (€M)', ytitle='Expected Assists (xA)',
               atitle='total_xA', ayaxis='player_market_value_euro')

# Prepping scatter function for xG + xA
yticks = np.arange(0, 20.5, 2)
xticks = np.arange(0, max(val) + 1000000, 1000000)
x_tick_labels = [str(int(tick / 1000000)) for tick in xticks]
cbar_ticks = np.arange(0, 20, 2)
cbar_label = 'Combined Expected Goals and Assists (xG + xA)'
# Scatter for xG + xA vs value
player_scatter(val, combined, combined, top_10_players_com,
               yticks, xticks, x_tick_labels,
               cbar_ticks, cbar_label,
               title="Top 5 leagues: combined xG and xA vs Value", xtitle='Transfer value (€M)', ytitle='Combined Expected Goals and Assists (xG + xA)',
               atitle='xG xA combined', ayaxis='player_market_value_euro')


# Radar - xG, xg vs goals, shots attempted, xG assisted, chances created, crosses, take ons attempted
def radar_data_setup(player_stats: pd.DataFrame, player: str, df_col, int_c1, int_c2, int_c3):
    """ #todo """
    #
    player: pd.DataFrame = player_stats.loc[player_stats["player_name_x"] == player]
    player = player[df_col]
    player_viz= player.columns[:]
    player_val = [round(player[column].iloc[0], 2) for column in player_viz]
    combined_viz = player_stats.fillna(0, inplace=True)
    player_percentiles = [int(stats.percentileofscore(
        player_stats[column], player[column].iloc[0])) for column in player_viz]
    slice_colors= ["#138015"] * int_c1 + ["#FFD449"] * int_c2
    text_colors = ["#000000"] * int_c3
    return player_percentiles, slice_colors, text_colors

df_col_tot = ['total_goals', 'xg_total', 'total_shots', 'total_take_ons',
          'total_carries', 'total_crosses_attempted', 'total_assists', 'total_key_passes', 'total_xA']

df_col_p90 = ['xg_p90', 'xg_p90', 'shots_p90', 'take_ons_p90',
          'carries_p90', 'crosses_attempted_p90', 'assists_p90', 'key_passes_p90', 'xA_p90']

att_names_tot = ["Non-Penalty Goals", "xG", "Shots", "Take-Ons",
             "Carries", "Crosses Attempted", "Assists", "Key Passes", "xA (Expected Assists)"]

att_names_p90 = ["Non-Penalty Goals P90", "xG P90", "Shots P90", "Take-Ons P90",
             "Carries P90", "Crosses Attempted P90", "Assists P90", "Key Passes P90", "xA (Expected Assists) P90"]

def radar_plot(params, data, s_colors, t_colors, title, subtitle):
    baker = PyPizza(
        params=params,
        background_color="#bebfc4",
        straight_line_color="#000000",
        straight_line_lw=1,
        last_circle_color="#000000",
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=20
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        data,
        figsize=(8, 8.5),
        color_blank_space="same",
        slice_colors=s_colors,
        value_colors=t_colors,
        value_bck_colors=s_colors,
        blank_alpha=0.4,
        kwargs_slices=dict(
            edgecolor="#000000", zorder=2, linewidth=1
        ),
        kwargs_params=dict(
            color="#000000", fontsize=11, va="center"
        ),
        kwargs_values=dict(
            color="#000000", fontsize=11, zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
            )
        )
    )

    # add title
    fig.text(
        0.51, 0.972, title, size=18,
        ha="center", weight='bold', color="#000000"
    )

    # add subtitle
    fig.text(
        0.01, 0.82,
        subtitle,
        size=13,
        ha="left", color="#000000"
    )

    sub = "Viz by Josh Mann"

    fig.text(
        0.5, 0.02, "Percentile comparison vs LM+LWs valued under €10m during the 2015/16 season (acc Transfermarkt)", size=13,
        color="#000000", ha="center"
    )

    fig.text(
        0.99, 0.02, f"{sub}", size=9, color="#000000",
        ha="right"
    )

    # add text
    fig.text(
        0.409, 0.94, "Attacking- Shooting", size=13,
        color="#000000"
    )

    fig.text(
        0.56, 0.94, "Attacking- Playmaking", size=13,
        color="#000000"
    )

    # add rectangles
    fig.patches.extend([
        plt.Rectangle(
            (0.38, 0.938), 0.025, 0.021, fill=True, color="#138015",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.532, 0.938), 0.025, 0.021, fill=True, color="#FFD449",
            transform=fig.transFigure, figure=fig
        ),
    ])
    # add logo
    logo = plt.imread('../data_logos/jmann logo.png')
    ax_image = add_image(
        logo, fig, left=0.430, bottom=0.419, width=0.165, height=0.152)

Kostic = f"Age- 24\nPosition- LM\nCurrent club- Stuttgart\nNationality- Serbian\nTransfer Valuation- €8m\nContract expiry- June 2019\nFormations played- 442/4231/4141/433\nMajor injuries- Torn muscle fiber (out for 35 days Sept-Nov 2015)"
Brady = f"Age- 24\nPosition- LM\nCurrent club- Norwich\nNationality- Irish\nTransfer Valuation- €10m\nContract expiry- July 2018\nFormations played- 4411/4231\nMajor injuries- Groin Injury (Out for 134 days Feb-Jun 2014)"
Samperio = f"Age- 23\nPosition- LW\nCurrent club- Mainz\nNationality- Spanish\nTransfer Valuation- €4m\nContract expiry- Aug 2018\nFormations played- 4231\nMajor injuries- None"
Benezet = f"Age- 25\nPosition- LW\nCurrent club- EA Guingamp\nNationality- French\nTransfer Valuation- €2m\nContract expiry- Jul 2018\nFormations played- 442\nMajor injuries- Adductor injury (Feb 2016-Current)"

# Kostic
player_percentiles, slice_colors, text_colors = radar_data_setup(
    players_xg, player="Filip Kostic", df_col=df_col_tot, int_c1=3, int_c2=6, int_c3=9)
radar_plot(att_names_tot, player_percentiles, slice_colors, text_colors, title="Filip Kostic", subtitle=Kostic)
# Brady
player_percentiles, slice_colors, text_colors = radar_data_setup(
    players_xg, player="Robbie Brady", df_col=df_col_tot, int_c1=3, int_c2=6, int_c3=9)
radar_plot(att_names_tot, player_percentiles, slice_colors, text_colors, title="Robbie Brady", subtitle=Brady)
# Samperio
player_percentiles, slice_colors, text_colors = radar_data_setup(
    players_xg, player="Jairo Samperio", df_col=df_col_tot, int_c1=3, int_c2=6, int_c3=9)
# Benezet
radar_plot(att_names_tot, player_percentiles, slice_colors, text_colors, title="Jairo Samperio", subtitle=Samperio)
player_percentiles, slice_colors, text_colors = radar_data_setup(
    players_xg, player="Nicolas Benezet", df_col=df_col_tot, int_c1=3, int_c2=6, int_c3=9)
radar_plot(att_names_tot, player_percentiles, slice_colors, text_colors, title="Nicolas Benezet", subtitle=Benezet)

plt.show()
