import numpy as np
from typing import List
from pandas import DataFrame as df
from datetime import date

from data_handling import add_new_teams,mean_new_teams
from math_handler import poly_3,prob_poiss

def p(elo_1, elo_2):
    return 1 / (10 ** (-(elo_1 - elo_2) / 400) + 1)


def margin_victory_multiplier(score_w, score_l):
    return np.log(score_w - score_l + 1) * 2.2 / ((score_w - score_l) * 0.001 + 2.2)


def total_goals_factor(score_w, score_l):
    return 100 / (100 + score_w + score_l)


k_factor = 20

def compute_elo_change(teams, game, table=None, param_lam=None):
    home_team = game[0]
    away_team = game[1]

    elo_h = teams.loc[teams['teams'] == home_team]['elo'].values[0]
    elo_a = teams.loc[teams['teams'] == away_team]['elo'].values[0]

    prob = p(elo_h, elo_a)

    if param_lam is None:
        home_score = game[2]
        away_score = game[3]
    else:
        lam_h = poly_3(prob, *param_lam)
        lam_a = poly_3(1 - prob, *param_lam)

        lam_h = lam_h if lam_h > 0 else 0
        lam_a = lam_a if lam_a > 0 else 0

        home_score, away_score = prob_poiss(lam_h, lam_a, 1)

    if home_score > away_score:
        point = 1
    elif home_score < away_score:
        point = 0
    else:
        point = 0.5

    f_delta = point - prob
    w_goals = home_score if home_score > away_score else away_score
    l_goals = home_score if home_score < away_score else away_score

    margin = margin_victory_multiplier(w_goals, l_goals)

    change = k_factor * f_delta * margin * total_goals_factor(w_goals, l_goals)

    teams.loc[teams.teams == home_team, 'elo'] = elo_h + change
    teams.loc[teams.teams == away_team, 'elo'] = elo_a - change

    if table is not None:
        h_points = 3 if home_score > away_score else 1 if home_score == away_score else 0
        a_points = 3 if home_score < away_score else 1 if home_score == away_score else 0

        table.loc[table.teams == home_team, 'points'] += h_points
        table.loc[table.teams == away_team, 'points'] += a_points
        return teams, prob, table
    else:
        return teams, prob

def append_elo_change(game, elo_change_dict, teams):
    for i in [0,1]:
        val = teams.loc[teams.teams == game[i], 'elo'].values[0]
        elo_change_dict[game[i]].append(val)

    return elo_change_dict

def create_elo_dict(teams,season_year):
    elo_dict = {}
    for team in teams.iterrows():
        elo_dict[team[1].teams] = [team[1].elo]
    return elo_dict


def compute_history(res_dict, teams, skip_years : List[int] = None):
    p_list = []
    mean_list = [1300]
    elo_history= {}
    for season_year, items in res_dict.items():
        # skip last season
        if skip_years is not None and season_year in skip_years:
            continue

        teams, new_teams = add_new_teams(teams, items["teams"],mean_list)
        elo_change = create_elo_dict(teams,season_year)
        for game in items["games"]:
            teams, prob = compute_elo_change(teams, game)
            elo_change = append_elo_change(game,elo_change,teams)

            # only get probabilities from seasons except for the first
            if season_year != min(res_dict.keys()):
                p_list.append((prob, game[2]))  # home goals
                p_list.append((1 - prob, game[3]))  # away goals

        #elo_change = df.from_dict(elo_change)
        #elo_change.insert(0, "Gameday", (np.array(elo_change.index.values.tolist())).tolist(), True)
        elo_history[season_year] = elo_change
        mean_list.append(mean_new_teams(teams, new_teams))
        teams.elo -= (teams.elo - np.mean(teams.elo)) / 3
        print(f"\n{season_year}/{season_year + 1}")
        print(teams.sort_values(by=['elo'], ascending=False))

    return teams,np.array(p_list).T,mean_list,elo_history

def create_initial_elo():
    teams = df.from_dict({
        'teams': [],
        'elo': []
    })
    return teams