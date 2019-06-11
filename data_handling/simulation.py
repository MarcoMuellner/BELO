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
import os
from typing import Dict
from operator import sub
import matplotlib.image as image

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

def plot_ponits(res_dict):
    fig_points, ax_list_points = pl.subplots(3, 6, figsize=(18, 10), sharex=True, sharey=True)
    fig_points: Figure

    fig_points.suptitle("Points distribution")

    ax_list_points: List[Axes]

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

        vals, num = np.unique(values['points'], return_counts=True)
        num = num / np.sum(num)

        ax_list_points[x, y].bar(vals, num, color='red')
        ax_list_points[x, y].set_xlim(0, 102)
        ax_list_points[x, y].set_title(team)

    fig_points.tight_layout()
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

def get_aspect(ax):
    width, height = ax.get_figure().get_size_inches()
    shit1, shit2, w, h = ax.get_position().bounds
    disp_ratio = (height * h) / (width * w)
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio

def plot_elo_history(res_dict):
    pl.rcParams.update(
        {'font.size': 22, 'axes.labelsize': 'xx-large', 'axes.titlesize': 'xx-large', 'xtick.labelsize': 'x-large',
         'ytick.labelsize': 'x-large', 'xtick.major.size': 20, 'xtick.major.width': 4, 'ytick.major.size': 20,
         'ytick.major.width': 4, 'ytick.minor.size': 5, 'ytick.minor.width': 1, 'axes.linewidth': 4.5, })

    games = range(1, 35)

    facecolor = 'black'
    edgecolor = 'black'

    c = 0
    for i, (years, data) in enumerate(res_dict.items()):
        for k in range(1, 35):
            c = c + 1
            fig, ax = pl.subplots(1, figsize=(36, 22))
            fig: Figure
            ax: Axes
            for j, (team, elos) in enumerate(data.items()):
                if team in ['Bayern Munich', 'FC Union Berlin', 'Fortuna Dusseldorf', '1. FC Koeln', 'Stuttgart',
                            'RB Leipzig',
                            'Nurnberg', 'Mainz', 'Dusseldorf', 'Ein Frankfurt', 'Leverkusen', 'Augsburg',
                            'Kaiserslautern']:
                    facecolor = 'red'
                elif team in ['Wolfsburg', 'Werder Bremen', 'Hannover']:
                    facecolor = 'green'
                elif team in ['Schalke 04', 'SC Paderborn', 'Hoffenheim', 'Hertha', 'Bochum', 'Darmstadt',
                              'Braunschweig',
                              'Hamburg']:
                    facecolor = 'Blue'
                elif team in ['Dortmund']:
                    facecolor = 'yellow'
                elif team in ['St Pauli']:
                    facecolor = 'brown'
                ax.set_ylim(1150, 1650)
                ax.set_title(f'Bundesliga season {years}/{int(years)+1}')
                ax.set_xlabel('Matchday')
                ax.set_ylabel('BElo')
                ax.set_xlim(0, 35)
                ax.set_xticks(range(1, 35))
                ax.plot(games[:k], elos[:k], '-', color=facecolor, lw=5, alpha=0.3, zorder=0)
                ax.plot(games[:k], elos[:k], '-', color=edgecolor, lw=7, alpha=0.3, zorder=0)
                im = image.imread(f'logos/{str(team)}.png')
                ratio = get_aspect(ax)
                ax.imshow(im, aspect='auto', extent=(
                games[k - 1] - 0.5, games[k - 1] + 0.5, elos[k - 1] - 0.5 / ratio, elos[k - 1] + 0.5 / ratio), zorder=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            pl.tight_layout()
            string = format(c, '05d')
            try:
                os.mkdir('history_data/elo_hist_plots')
            except FileExistsError:
                pass
            fig.savefig(f'history_data/elo_hist_plots/{string}.png')
            pl.close()
            print('season',years, f'/{int(years)+1}','matchday', k,'/34\n')

def position(posi):
    x = int((posi)/6)
    y = int((posi)%6)
    return x, y

def plot_result(res_dict):



    pl.rcParams.update(
        {'font.size': 22, 'axes.labelsize': 'large', 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large',
         'xtick.major.size': 20, 'xtick.major.width': 4, 'ytick.major.size': 20, 'ytick.major.width': 4,
         'ytick.minor.size': 5, 'ytick.minor.width': 1, 'axes.linewidth': 4.5, })

    fig_pos, ax_list_pos = pl.subplots(3, 6, figsize=(36, 22), sharex=True)
    fig_pie, ax_list_pie = pl.subplots(3, 6, figsize=(36, 22), sharex=True)

    fig_pos: Figure
    fig_pie: Figure

    ax_list_pos : List[Axes]
    ax_list_pie : List[Axes]

    listteams = []
    listpos = []
    for i, (team, values) in enumerate(res_dict.items()):
        listteams.append(team)
        listpos.append(np.mean(values['pos']))

    sortlist = np.argsort(listpos)
    listteams = np.array(listteams)[sortlist]

    for i, (team, values) in enumerate(res_dict.items()):

        for k in range(len(listteams)):
            if team == listteams[k]:
                posi = k
                break

        #xs = [0,1,2,0,2,0,1,0,1,2,2,0,0,2,1,2,1,1]
        #ys = [0,2,1,1,2,2,4,5,1,0,5,3,4,4,5,3,3,0]#

        #x = xs[i]
        #y = ys[i]
        x,y = position(posi)
        vals, num = np.unique(values['pos'], return_counts=True)
        num = num / np.sum(num)

        downs = []
        try:
            downs.append(num[15] * 100)
        except:
            pass
        try:
            downs.append(num[16] * 100)
        except:
            pass
        try:
            downs.append(num[17] * 100)
        except:
            pass

        pies = [num[0], np.sum(num[1:4]), np.sum(num[4:7]), np.sum(downs) / 100, np.sum(num[7:15])]


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


        ax_list_pos[x, y].bar(vals, num, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw)
        wedges, text = ax_list_pie[x, y].pie(pies, colors=['green', 'blue', 'orange', 'red', 'gray'])
        pl.setp(wedges, width=0.50)
        for w in range(len(wedges)):
            wedges[w].set_linewidth(2)
            wedges[w].set_edgecolor('white')
            ###### compare to real results ####
            '''
            if str(team) == 'Bayern Munich':
                if w != 0:
                    wedges[w].set_alpha(0.3)
            elif str(team) in ['Dortmund','RB Leipzig', 'Leverkusen' ]:
                if w != 1:
                    wedges[w].set_alpha(0.3)
            elif str(team) in ['Wolfsburg','Ein Frankfurt', "M'gladbach" ]:
                if w != 2:
                    wedges[w].set_alpha(0.3)
            elif str(team) in ['Nurnberg','Hannover', 'Stuttgart' ]:
                if w != 3:
                    wedges[w].set_alpha(0.3)
            else:
                if w != 4:
                    wedges[w].set_alpha(0.3)
            '''
        im = image.imread(f'logos/{str(team)}.png')
        ax_list_pie[x, y].imshow(im, aspect='auto', extent=(-0.5, 0.5, -0.5, 0.5), zorder=11)
        ax_list_pos[x, y].set_xlim(0, 18.5)
        if str(team) == 'Bayern Munich':
            ax_list_pos[x, y].set_ylim(0, 0.9)
            ratio = get_aspect(ax_list_pos[x, y])
            im = image.imread(f'logos/{str(team)}.png')
            ax_list_pos[x, y].imshow(im, aspect='auto', extent=(5, 14, 0.7 - 9 / ratio, 0.7), zorder=11)
            ax_list_pos[x, y].set_yticks([0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90])
            im = image.imread(f'logos/Schale.png')
            ax_list_pie[x, y].imshow(im, aspect='auto', extent=(-1.2, -0.7, 0.7, 1.2), zorder=11)
            ax_list_pie[x, y].plot([-0.95, -0.5], [0.95, 0.5], 'k-', lw=3, zorder=2)
            ax_list_pie[x, y].plot([-0.95, -0.5], [0.95, 0.5], 'w-', lw=5, zorder=1)
        else:
            ax_list_pos[x, y].set_ylim(0, 0.3)
            im = image.imread(f'logos/{str(team)}.png')
            ratio = get_aspect(ax_list_pos[x, y])
            ax_list_pos[x, y].imshow(im, aspect='auto', extent=(5, 14, 0.23 - 9 / ratio, 0.23), zorder=11)
        if str(team) == 'Dortmund':
            im = image.imread(f'logos/cltrophy.png')
            ax_list_pie[x, y].imshow(im, aspect='auto', extent=(-1.2, -0.6, 0.6, 1.2), zorder=11)
            ax_list_pie[x, y].plot([-0.95, -0.5], [0.95, 0.5], 'k-', lw=3, zorder=2)
            ax_list_pie[x, y].plot([-0.95, -0.5], [0.95, 0.5], 'w-', lw=5, zorder=1)
        if str(team) == 'RB Leipzig':
            im = image.imread(f'logos/eltrophy.png')
            ax_list_pie[x, y].imshow(im, aspect='auto', extent=(-1.2, -0.6, -1.2, -0.6), zorder=11)
            ax_list_pie[x, y].plot([-0.9, -0.4], [-0.95, -0.6], 'k-', lw=3, zorder=2)
            ax_list_pie[x, y].plot([-0.9, -0.4], [-0.95, -0.6], 'w-', lw=5, zorder=1)
        if str(team) == 'Leverkusen':
            im = image.imread(f'logos/buli.png')
            ax_list_pie[x, y].imshow(im, aspect='auto', extent=(0.7, 1.2, -1.2, -0.7), zorder=11)
            ax_list_pie[x, y].plot([1, 0.5], [-0.9, -0.5], 'k-', lw=3, zorder=2)
            ax_list_pie[x, y].plot([1, 0.5], [-0.9, -0.5], 'w-', lw=5, zorder=1)
        if str(team) == 'Fortuna Dusseldorf':
            im = image.imread(f'logos/abstieg.png')
            ax_list_pie[x, y].imshow(im, aspect='auto', extent=(-1.2, -0.7, 0.7, 1.2), zorder=11)
            ax_list_pie[x, y].plot([-0.8, -0.3], [0.9, 0.65], 'k-', lw=3, zorder=2)
            ax_list_pie[x, y].plot([-0.8, -0.3], [0.9, 0.65], 'w-', lw=5, zorder=1)
        if y in [0] or (y==1 and x == 0):
            ax_list_pos[x, y].spines["top"].set_visible(False)
            ax_list_pos[x, y].spines["right"].set_visible(False)
            ax_list_pos[x, y].spines["bottom"].set_visible(False)
            ax_list_pos[x, y].set_xticks([])
        else:
            ax_list_pos[x, y].spines["top"].set_visible(False)
            ax_list_pos[x, y].spines["right"].set_visible(False)
            ax_list_pos[x, y].spines["bottom"].set_visible(False)
            ax_list_pos[x, y].spines["left"].set_visible(False)
            ax_list_pos[x, y].set_xticks([])
            ax_list_pos[x, y].set_yticks([])

    fig_pos.tight_layout(h_pad=1, w_pad=0.2)
    fig_pie.tight_layout(h_pad=1, w_pad=0.2)
    fig_pos.savefig('position_2020.png')
    fig_pie.savefig('pies_2020.png')
    #pl.show()