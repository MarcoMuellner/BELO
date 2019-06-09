from data_handling import get_dict,add_2019_season,simulate_season,plot_result,create_elo_frames,plot_elo_history
from elo_computation import create_initial_elo,compute_history
from math_handler import create_poisson_distributions
import json
from pandas import read_csv
import numpy as np

use_history=True
use_simu_history=True

data = get_dict()
data = add_2019_season(data)
if not use_history:
    teams = create_initial_elo()
    teams,p_list,mean_list,elo_history = compute_history(data,teams,[2019])
    with open("history_data/elo_history.json",'w') as f:
        json.dump(elo_history,f,indent=4)
    teams.to_csv("history_data/teams.csv")
    np.savetxt("history_data/p_list.txt",p_list)
    np.savetxt("history_data/mean_list.txt",np.array(mean_list))
else:
    teams = read_csv("history_data/teams.csv")
    p_list = np.loadtxt("history_data/p_list.txt")
    mean_list = np.loadtxt("history_data/mean_list.txt").tolist()
    with open("history_data/elo_history.json",'r') as f:
        elo_history = json.load(f)

elo_history = create_elo_frames(elo_history)
plot_elo_history(elo_history,2018)

if not use_simu_history:
    popt_poly = create_poisson_distributions(p_list,False)
    simu_result, table_lists = simulate_season(data,2019,teams,popt_poly,mean_list,50000)

    with open("history_data/simu_results.json",'w') as f:
        json.dump(simu_result,f,indent=4)
else:
    with open("history_data/simu_results.json",'r') as f:
        simu_result = json.load(f)

plot_result(simu_result)
