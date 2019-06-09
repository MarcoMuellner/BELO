import numpy as np
from datapackage import Package
from pandas import DataFrame as df


def get_data():
    package = Package('https://datahub.io/sports-data/german-bundesliga/datapackage.json')
    data = []

    for i, resource in enumerate(reversed(package.resources)):
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            data.append(resource.read())

    return data


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
        res_dict[season_year]["games"] = np.array((arr_s[i + 1], arr_s[i + 2], arr_s[i + 3], arr_s[i + 4],arr_s[i])).T.tolist()
    return res_dict


def add_new_teams(teams, teams_season, mean_list):
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

def create_table(teams):
    table = {
        'teams': teams.teams.tolist(),
        'points': np.zeros(len(teams.teams.tolist()))
    }
    return df.from_dict(table)

def create_elo_frames(elo_history):
    ret_dict = {}
    for year,elo_dict in elo_history.items():
        ret_dict[year] = df.from_dict(elo_dict)
        ret_dict[year].insert(0, "Gameday", (np.array(ret_dict[year].index.values.tolist())).tolist(), True)
    return ret_dict

def create_history(teams):
    res_dict = {}
    for team in teams.iterrows():
        res_dict[team[1].teams] = [team[1].elo]
    return res_dict

def add_history(history,teams):
    for team in teams.iterrows():
        history[team[1].teams].append(team[1].elo)
    return history

def get_dict():
    return split_up_data(get_data())