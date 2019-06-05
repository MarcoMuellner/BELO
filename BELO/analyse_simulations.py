from pandas import DataFrame as df
import os
import matplotlib.pyplot as pl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List
res_dict = {}

for _,_,files in os.walk("simu_results"):
    for file in files:
        frame = df.from_csv(f"simu_results/{file}")
        for i in frame.iterrows():
            team = i[1]['teams']
            pos = i[1]['Position']
            points = i[1]['points']

            if team in res_dict.keys():
                res_dict[team]['pos'].append(pos)
                res_dict[team]['points'].append(points)
            else:
                res_dict[team] = {
                    'pos':[pos],
                    'points':[points]
                }

fig_pos,ax_list_pos = pl.subplots(3,6,figsize=(18,10),sharex=True)
fig_points,ax_list_points = pl.subplots(3,6,figsize=(18,10),sharex=True)
fig_points : Figure
fig_pos : Figure

fig_points.suptitle("Points distribution")
fig_pos.suptitle("Position distribution")

ax_list_points : List[Axes]
ax_list_pos : List[Axes]

for i,(team, values) in enumerate(res_dict.items()):
    x = 0 if i < 6 else 1 if i < 12 else 2
    y = i if i < 6 else i - 6 if i < 12 else i - 12
    try:
        ax_list_pos[x,y].hist(values['pos'],density=True)
    except:
        pass
    ax_list_pos[x, y].set_title(team)
    ax_list_points[x, y].hist(values['points'], density=True)
    ax_list_points[x, y].set_xlim(0,102)
    ax_list_points[x, y].set_title(team)

pl.show()