from copy import deepcopy
from multiprocessing import Pool, cpu_count, Process
import numpy as np
from data_handling import add_new_teams,create_table,create_history,add_history,mean_new_teams
from elo_computation import compute_elo_change
from pandas import DataFrame as df
from typing import List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib.pyplot as pl
import time
from typing import Dict

def add_2019_season(data):
    season_2018 = data[2018]
    teams = deepcopy(season_2018['teams'])
    games = deepcopy(season_2018['games'])
    replace_list = {
        'Nurnberg' : "1. FC Koeln",
        'Hannover':'SC Paderborn',
        'Stuttgart':'FC Union Berlin'
    }

    for old_team,new_team in replace_list.items():
        teams.remove(old_team)
        teams.append(new_team)

    for game in games:
        if game[0] in replace_list.keys():
            game[0] = replace_list[game[0]]
        if game[1] in replace_list.keys():
            game[1] = replace_list[game[1]]

    data[2019] = {'teams':teams,'games':games}
    return data


def _simulate_season(res_dict, chosen_season, used_teams, param_lam,mean_list, n):
    for season_year, items in res_dict.items():
        # skip last season
        if season_year != chosen_season:
            continue

        used_teams, new_teams = add_new_teams(used_teams, items["teams"],mean_list)
        table = create_table(used_teams)

        history = create_history(used_teams)
        for game in items["games"]:
            now = time.time()
            used_teams, prob, table = compute_elo_change(used_teams, game, table, param_lam)
            history = add_history(history,used_teams)

        table = table.sort_values(by=['points'], ascending=False).reset_index(drop=True)
        table.insert(0, "Position", (np.array(table.index.values.tolist()) + 1).tolist(), True)

        print(f"\n{season_year}/{season_year + 1} --> simulation {n + 1} done")
    return table


def simulate_season(res_dict, chosen_season, teams, param_lam,mean_list, n):
    run_list = []
    nr_of_cores = cpu_count()
    for i in range(0,n+1):
        run_list.append((res_dict,chosen_season,deepcopy(teams),param_lam,mean_list,i))

    pool = Pool(processes=nr_of_cores)
    resulting_tables = pool.starmap(_simulate_season, run_list)
    return combine_results(resulting_tables),resulting_tables

def combine_results(tables : List[df]):
    res_dict = {}
    for table in tables:
        for i in table.iterrows():
            team = i[1]['teams']
            pos = i[1]['Position']
            points = int(i[1]['points'])

            if team in res_dict.keys():
                res_dict[team]['pos'].append(pos)
                res_dict[team]['points'].append(points)
            else:
                res_dict[team] = {
                    'pos': [pos],
                    'points': [points]
                }

    return res_dict

def plot_result(res_dict):
    fig_pos, ax_list_pos = pl.subplots(3, 6, figsize=(18, 10), sharex=True, sharey=True)
    fig_points, ax_list_points = pl.subplots(3, 6, figsize=(18, 10), sharex=True, sharey=True)
    fig_points: Figure
    fig_pos: Figure

    fig_points.suptitle("Points distribution")
    fig_pos.suptitle("Position distribution")

    ax_list_points: List[Axes]
    ax_list_pos: List[Axes]

    table_dict = {
        "team": [],
        "winner_chance": [],
        "top_four_chance": [],
        "top_6_chance": [],
        "lower_3_chance": [],
    }

    for i, (team, values) in enumerate(res_dict.items()):
        x = 0 if i < 6 else 1 if i < 12 else 2
        y = i if i < 6 else i - 6 if i < 12 else i - 12

        vals, num = np.unique(values['pos'], return_counts=True)
        num = num / np.sum(num)

        ax_list_pos[x, y].bar(vals, num, color='green')
        ax_list_pos[x, y].set_title(team)

        table_dict["team"].append(team)
        table_dict["winner_chance"].append(np.round(num[0] * 100))
        table_dict["top_four_chance"].append(np.round(np.sum(num[0:4]) * 100))
        table_dict["top_6_chance"].append(np.round(np.sum(num[0:6]) * 100))
        table_dict["lower_3_chance"].append(np.round(np.sum(num[-4:-1]) * 100))

        vals, num = np.unique(values['points'], return_counts=True)
        num = num / np.sum(num)

        ax_list_points[x, y].bar(vals, num, color='red')
        ax_list_points[x, y].set_xlim(0, 102)
        ax_list_points[x, y].set_title(team)

    table_dict = df.from_dict(table_dict)
    fig_points.tight_layout()
    fig_pos.tight_layout()
    pl.show()

def get_color(team):
    facecolor = 'black'
    edgecolor = 'black'
    lw = 3.5
    if team in ['Bayern Munich', 'FC Union Berlin', 'Fortuna Dusseldorf', '1. FC Koeln', 'Stuttgart', 'RB Leipzig',
                'Nurnberg', 'Mainz', 'Dusseldorf', 'Ein Frankfurt', 'Leverkusen', 'Augsburg']:
        facecolor = 'red'
    elif team in ['Wolfsburg', 'Werder Bremen', 'Hannover']:
        facecolor = 'green'
    elif team in ['Schalke 04', 'SC Paderborn', 'Hoffenheim', 'Hertha']:
        facecolor = 'Blue'
    elif team in ['Dortmund']:
        facecolor = 'yellow'
    else:
        lw = 0.5
        edgecolor = 'white'

    return facecolor,edgecolor,lw