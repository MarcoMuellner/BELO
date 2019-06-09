from pandas import DataFrame as df
from pandas import read_csv
import os
import matplotlib.pyplot as pl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List
import numpy as np
import json
"""
res_dict = {}

for _,_,files in os.walk("simu_results"):
    for file in files:
        frame = read_csv(f"simu_results/{file}")
        for i in frame.iterrows():
            team = i[1]['teams']
            pos = i[1]['Position']
            points = int(i[1]['points'])

            if team in res_dict.keys():
                res_dict[team]['pos'].append(pos)
                res_dict[team]['points'].append(points)
            else:
                res_dict[team] = {
                    'pos':[pos],
                    'points':[points]
                }
"""
with open("league_statistics.json",'r') as f:
    res_dict = json.load(f)
    #json.dump(res_dict,f,indent=4)

#res_dict = read_csv("league_statistics.csv").to_dict()
#del res_dict['Unnamed: 0']

fig_pos,ax_list_pos = pl.subplots(3,6,figsize=(18,10),sharex=True,sharey=True)
fig_points,ax_list_points = pl.subplots(3,6,figsize=(18,10),sharex=True,sharey=True)
fig_points : Figure
fig_pos : Figure
fig_table : Figure = pl.figure(figsize=(10,14))

fig_points.suptitle("Points distribution")
fig_pos.suptitle("Position distribution")

ax_list_points : List[Axes]
ax_list_pos : List[Axes]
ax_table :Axes = fig_table.subplots()

table_dict = {
    "team":[],
    "winner_chance":[],
    "top_four_chance":[],
    "top_6_chance":[],
    "lower_3_chance":[],
}


for i,(team, values) in enumerate(res_dict.items()):
    x = 0 if i < 6 else 1 if i < 12 else 2
    y = i if i < 6 else i - 6 if i < 12 else i - 12

    vals, num = np.unique(values['pos'], return_counts=True)
    num = num / np.sum(num)

    ax_list_pos[x,y].bar(vals,num,color='green')
    ax_list_pos[x, y].set_title(team)

    table_dict["team"].append(team)
    table_dict["winner_chance"].append(np.round(num[0]*100))
    table_dict["top_four_chance"].append(np.round(np.sum(num[0:4]) * 100))
    table_dict["top_6_chance"].append(np.round(np.sum(num[0:6]) * 100))
    table_dict["lower_3_chance"].append(np.round(np.sum(num[-4:-1]) * 100))

    vals, num = np.unique(values['points'], return_counts=True)
    num = num / np.sum(num)


    ax_list_points[x,y].bar(vals,num,color='red')
    ax_list_points[x, y].set_xlim(0,102)
    ax_list_points[x, y].set_title(team)

table_dict = df.from_dict(table_dict)
print(table_dict.sort_values(by=['winner_chance','top_four_chance','top_6_chance'],ascending=False))
fig_points.tight_layout()
fig_pos.tight_layout()
pl.show()