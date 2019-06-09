from datapackage import Package
from pandas import read_csv
import numpy as np
from scipy.special import factorial
from pandas import DataFrame as df
import matplotlib.pyplot as pl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List
from scipy.optimize import curve_fit
from copy import deepcopy
from multiprocessing import Pool, cpu_count, Process


def p(elo_1, elo_2):
    return 1 / (10 ** (-(elo_1 - elo_2) / 400) + 1)


def margin_victory_multiplier(score_w, score_l):
    return np.log(score_w - score_l + 1) * 2.2 / ((score_w - score_l) * 0.001 + 2.2)


def total_goals_factor(score_w, score_l):
    return 100 / (100 + score_w + score_l)


def poisson(x, lam):
    return (lam ** x / factorial(x)) * np.exp(-lam)


def linear(x, a, b):
    return a + b * x


def poly_3(x, a, b, c, d, e):
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4


def prob_poiss(lam_1, lam_2, n):
    x_1 = np.random.poisson(lam_1, n)
    x_2 = np.random.poisson(lam_2, n)
    return int(np.round(np.mean(x_1))), int(np.round(np.mean(x_2)))


k_factor = 20


def get_data():
    package = Package('https://datahub.io/sports-data/german-bundesliga/datapackage.json')
    data = []

    for i, resource in enumerate(reversed(package.resources)):
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            data.append(resource.read())

    return data


def create_initial_elo(res_dict):
    teams = df.from_dict({
        'teams': [],
        'elo': []
    })
    return teams


def get_season(season, index):
    try:
        season_year = season[0][index].year
    except:
        season_year = int(season[0][index].split("/")[-1]) if len(season[0][index].split("/")[-1]) == 4 else int(
            f'20{season[0][index].split("/")[-1]}')

    return season_year


def split_up_data(data):
    # iterate over seasons
    res_dict = {}
    for season in data:
        # take first year as identifier for season
        i = 0
        try:
            season_year = get_season(season, i)
        except:
            i += 1
            season_year = get_season(season, i)

        res_dict[season_year] = {}
        arr_s = np.array(season).T
        res_dict[season_year]["teams"] = np.unique(arr_s[i + 1]).tolist()
        res_dict[season_year]["games"] = np.array((arr_s[i + 1], arr_s[i + 2], arr_s[i + 3], arr_s[i + 4])).T.tolist()
    return res_dict


mean_list = [1300]


def add_new_teams(teams, teams_season):
    new_teams = [i for i in teams_season if i not in teams['teams'].values]
    new_initial_elo = (np.zeros(len(new_teams)) + np.mean(mean_list)).tolist()
    old_teams = [i[1]['teams'] for i in teams.iterrows() if i[1]['teams'] in teams_season]
    old_elo = [i[1]['elo'] for i in teams.iterrows() if i[1]['teams'] in teams_season]
    teams = df.from_dict({
        'teams': old_teams + new_teams,
        'elo': old_elo + new_initial_elo
    })
    return teams, new_teams


def mean_new_teams(teams, new_teams):
    val = []
    for team in new_teams:
        val.append(teams.loc[teams.teams == team].elo.values[0])

    return np.mean(val)


def create_poisson_distributions(p_list):
    fig, ax_list = pl.subplots(2, 4, figsize=(16, 10), sharex=True)

    fig: Figure
    ax_list: List[Axes]

    p_range = np.array((np.arange(0.11, 0.91, 0.1), np.arange(0.21, 1.01, 0.1))).T
    lam_list = []

    for i, (p_l, p_u) in enumerate(p_range):
        mask = np.logical_and(p_list[0] > p_l, p_list[0] < p_u)
        vals, num = np.unique(p_list[1][mask], return_counts=True)

        while len(vals) <= 9:
            vals = np.array(vals.tolist() + [max(vals) + 1])
            num = np.array(num.tolist() + [0])

        num = num / np.sum(num)
        median_p = np.median(p_list[0][mask])
        # fit poisson
        popt, pcov = curve_fit(poisson, vals, num)
        perr = np.sqrt(np.diag(pcov))

        x_plot = np.linspace(0, max(vals), num=500)

        x = 0 if i < 4 else 1
        y = i if i < 4 else i - 4
        ax_list[x, y].bar(vals, num, alpha=0.5, label='Goal distribution')
        ax_list[x, y].set_title('%.2f' % median_p)
        ax_list[x, y].plot(x_plot, poisson(x_plot, *popt), color='red', label='fit')
        ax_list[x, y].legend()

        lam_list.append((median_p, popt[0], perr[0]))

    arr_l = np.array(lam_list).T
    popt_poly, pcov_poly = curve_fit(poly_3, arr_l[0], arr_l[1], sigma=arr_l[2])
    x_plot = np.linspace(min(arr_l[0]), max(arr_l[0]), num=500)
    pl.close()

    pl.figure(figsize=(16, 10))
    pl.errorbar(arr_l[0], arr_l[1], yerr=arr_l[2], fmt='o')
    pl.plot(x_plot, poly_3(x_plot, *popt_poly))
    pl.title(r"$\lambda$ value distribution")
    pl.close()
    return popt_poly


def create_table(teams):
    table = {
        'teams': teams.teams.tolist(),
        'points': np.zeros(len(teams.teams.tolist()))
    }
    return df.from_dict(table)

def create_history(teams):
    res_dict = {}
    for team in teams.iterrows():
        res_dict[team[1].teams] = [team[1].elo]
    return res_dict

def add_history(history,teams):
    for team in teams.iterrows():
        history[team[1].teams].append(team[1].elo)
    return history


def _simulate_season(res_dict, chosen_season, used_teams, param_lam, n, plot=False):
    for season_year, items in res_dict.items():
        # skip last season
        if season_year != chosen_season:
            continue

        used_teams, new_teams = add_new_teams(used_teams, items["teams"])
        table = create_table(used_teams)

        if plot:
            colors = read_csv("colors.csv")
            fig: Figure = pl.figure()
            ax: Axes = fig.subplots()

        history = create_history(used_teams)
        for game in items["games"]:
            used_teams, prob, table = compute_elo_change(used_teams, game, table, param_lam)
            history = add_history(history,used_teams)
            if plot:
                plot_elo(used_teams,history, colors, ax)

        mean_list.append(mean_new_teams(used_teams, new_teams))
        used_teams.elo -= (used_teams.elo - np.mean(used_teams.elo)) / 3
        table = table.sort_values(by=['points'], ascending=False).reset_index(drop=True)
        table.insert(0, "Position", (np.array(table.index.values.tolist()) + 1).tolist(), True)

        print(f"\n{season_year}/{season_year + 1} --> simulation {n + 1}")
        print(table)
        # tables.append(table)
        table.to_csv(f"simu_results/table_{n}.csv")


def simulate_season(res_dict, chosen_season, teams, param_lam, n):
    run_list = []
    # nr_of_cores = cpu_count()
    # for i in range(0,n+1):
    #    run_list.append((res_dict,chosen_season,deepcopy(teams),param_lam,i,True))

    # pool = Pool(processes=nr_of_cores)
    # pool.starmap(_simulate_season, run_list)
    _simulate_season(res_dict, chosen_season, deepcopy(teams), param_lam, n, True)


def plot_elo(teams,team_history, colors, ax: Axes):
    for team,values in team_history.items():
        color = colors.loc[colors['teams'] == team]
        ax.plot(values,color=(int(color.r.values)/255,int(color.g.values)/255,int(color.b.values)/255,1))
        #ax.plot(values[-1],marker='o', color=(int(color.r.values) / 255, int(color.g.values) / 255, int(color.b.values) / 255))
    pl.draw()
    pl.pause(0.3)


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


res_dict = split_up_data(get_data())
"""
teams = create_initial_elo(res_dict)
p_list = []
for season_year,items in res_dict.items():
    #skip last season
    if season_year == 2018:
        continue

    teams,new_teams = add_new_teams(teams,items["teams"])
    for game in items["games"]:
        teams,prob = compute_elo_change(teams,game)

        #only get probabilities from seasons except for the first
        if season_year != min(res_dict.keys()):
            p_list.append((prob, game[2])) #home goals
            p_list.append((1 - prob, game[3])) # away goals

    mean_list.append(mean_new_teams(teams,new_teams))
    teams.elo -= (teams.elo - np.mean(teams.elo))/3
    print(f"\n{season_year}/{season_year+1}")
    print(teams.sort_values(by=['elo'],ascending=False))

teams.to_csv("teams.csv")
p_list = np.array(p_list).T
"""
# np.savetxt('prob.txt',p_list)
p_list = np.loadtxt('prob.txt')
teams = df.from_csv("teams.csv")
param_lam = create_poisson_distributions(p_list)

simulate_season(res_dict, 2018, teams, param_lam, 1)
