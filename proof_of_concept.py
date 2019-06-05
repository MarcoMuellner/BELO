from builtins import reversed

from datapackage import Package
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import factorial
from scipy.optimize import curve_fit


def p(elo_1,elo_2):
    return 1/(10**(-(elo_1-elo_2)/400) + 1)

def margin_victory_multiplier(score_w,score_l):
    return np.log(score_w-score_l + 1)*2.2/((score_w-score_l)*0.001 +2.2)

def total_goals_factor(score_w,score_l):
    return 100/(100 + score_w + score_l)

def poisson(x,lam):
    return (lam**x/factorial(x))*np.exp(-lam)

def linear(x,a,b):
    return a+b*x

def poly_3(x,a,b,c,d,e):
    return a+b*x+c*x**2+d*x**3+e*x**4

def prob_poiss(lam_1,lam_2,n):
    x_1 = np.random.poisson(lam_1,n)
    x_2 = np.random.poisson(lam_2,n)
    return int(np.round(np.mean(x_1))),int(np.round(np.mean(x_2)))
k_factor = 20


package = Package('https://datahub.io/sports-data/german-bundesliga/datapackage.json')

# print list of all resources:
print(package.resource_names)

# print processed tabular data (if exists any)
data = []

for i,resource in enumerate(reversed(package.resources)):
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        data.append(resource.read())

season_2008 = np.array(data[0]).T
teams = np.unique(season_2008[2])
initial_elo = np.zeros(len(teams)) + 1300
teams = np.vstack((teams,initial_elo))
y_pos = np.arange(len(teams[0]))

fig, ax = pl.subplots(figsize=(16,10))
line = ax.barh(y_pos, teams[1],  align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(teams[0])
pl.draw()
#pl.pause(1)

mean_list = [1300]
p_list = []
f_list = []
for j in data[0:-1]:
    season = np.array(j).T

    teams_season = np.unique(season[2])

    new_teams = [i for i in teams_season if i not in teams[0]]
    new_initial_elo = np.zeros(len(new_teams)) + np.mean(mean_list)

    old_teams = np.array([i for i in teams.T if i[0] in teams_season]).T
    teams = np.hstack((old_teams,np.array((new_teams,new_initial_elo))))

    for i in teams.T:
        i[1] = float(i[1])

    line.remove()
    line = ax.barh(y_pos, teams[1], align='center',
                   color='green', ecolor='black')
    ax.set_yticklabels(teams[0])

    fig.canvas.draw()
    fig.canvas.flush_events()
    #pl.pause(2)

    for n,i in enumerate(season.T):
        id_h = np.where(teams[0] == i[2])
        id_a = np.where(teams[0] == i[3])

        elo_h = teams[1][id_h]
        elo_a = teams[1][id_a]

        prob = p(elo_h,elo_a)

        if i[4] > i[5]:
            point = 1
        elif i[4] < i[5]:
            point = 0
        else:
            point = 0.5

        f_delta = point - prob
        f_list.append((prob,point))

        w_goals = i[4] if i[4] > i[5] else i[5]
        l_goals = i[4] if i[4] < i[5] else i[5]

        p_list.append((prob,i[4]))
        p_list.append((1-prob, i[5]))

        margin = margin_victory_multiplier(w_goals,l_goals)

        if np.abs(f_delta) > 0.7:
            print(i)

        change = k_factor*f_delta*margin*total_goals_factor(w_goals,l_goals)

        teams[1][id_h] += change
        teams[1][id_a] -= change

        if n%20 == 0:
            line.remove()
            line = ax.barh(y_pos, teams[1], align='center',
                           color='green', ecolor='black')
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.suptitle(f"{i[1]}")
            #pl.pause(0.03)

    """

    """

    val = 0
    cnt = 0
    for i in teams.T:
        if i[0] in new_teams:
            #print(i)
            cnt +=1
            val += i[1]

    if cnt == 0:
        val = 1300
        cnt = 1
    mean_list.append(val/cnt)

    for i in teams.T:
        i[1] -= (i[1] - np.mean(i[1]))/2
    print("\n")
    arg_id = np.argsort(teams[1])
    for n,i in enumerate(zip(reversed(teams[0][arg_id]),reversed(teams[1][arg_id]))):
        print(f"{n+1}.: {i[0]} -> {'%.2f' % i[1]}")

pl.close(fig)
arr = np.array(p_list).T
arr_2 = np.array(f_list).T

#p_list = [(0,0.33),(0.33,0.66),(0.66,0.99)]
#p_list = [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1)]
#p_list = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1)]
p_list = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1)]
lam_list = []
for (p_l, p_u) in p_list:
    fig_2 = pl.figure()
    ax = fig_2.add_subplot(111)
    mask = np.logical_and(arr[0] > p_l, arr[0] < p_u)
    vals, num = np.unique(arr[1][mask], return_counts=True)

    while len(vals) <= 10:
        vals = np.array(vals.tolist() + [max(vals)+1])
        num = np.array(num.tolist() + [0])

    try:
        num = num / np.sum(num)
    except:
        continue

    popt,pcov = curve_fit(poisson,vals,num)
    perr = np.sqrt(np.diag(pcov))
    x_plot = np.linspace(0,max(vals),num=500)
    ax.bar(vals, num, alpha=0.5)
    ax.plot(x_plot,poisson(x_plot,*popt))
    median_p = np.median(arr[0][mask])[0]
    ax.set_title('%.2f' % median_p)
    ax.set_ylim(0,0.5)
    lam_list.append((median_p,popt[0],perr[0]))
    pl.close(fig_2)

arr_l = np.array(lam_list).T
pl.figure()
pl.errorbar(arr_l[0],arr_l[1],yerr=arr_l[2],fmt='o')
popt_poly,pcov_poly = curve_fit(poly_3,arr_l[0],arr_l[1],sigma=arr_l[2])
pl.plot(arr_l[0],poly_3(arr_l[0],*popt_poly))

try:
    pl.legend()
    pl.close()
except:
    pass

p_list = [(0.23,0.27),(0.33,0.36),(0.42,0.45),(0.47,0.53),(0.53,0.58),(0.54,0.65),(0.85,0.88)]

for (p_l, p_u) in p_list:
    fig_2 = pl.figure()
    ax = fig_2.add_subplot(111)
    mask = np.logical_and(arr[0] > p_l, arr[0] < p_u)
    vals, num = np.unique(arr[1][mask], return_counts=True)

    while len(vals) <= 10:
        vals = np.array(vals.tolist() + [max(vals)+1])
        num = np.array(num.tolist() + [0])

    try:
        num = num / np.sum(num)
    except:
        continue

    ax.bar(vals, num, alpha=0.5)
    try:
        median_p = np.median(arr[0][mask])[0]
    except IndexError:
        median_p = np.median(arr[0][mask])
    ax.set_title('%.2f' % median_p)
    ax.set_ylim(0, 0.5)
    lam = poly_3(median_p,*popt_poly)
    ax.axvline(x=np.round(lam))
    pl.close()

try:
    pl.legend()
    pl.close()
except:
    pass

season = np.array(data[-1]).T

teams_season = np.unique(season[2])

new_teams = [i for i in teams_season if i not in teams[0]]
new_initial_elo = np.zeros(len(new_teams)) + 1300

old_teams = np.array([i for i in teams.T if i[0] in teams_season]).T
teams = np.hstack((old_teams,np.array((new_teams,new_initial_elo))))

for i in teams.T:
    i[1] = float(i[1])

line.remove()
line = ax.barh(y_pos, teams[1], align='center',
               color='green', ecolor='black')
ax.set_yticklabels(teams[0])

fig.canvas.draw()
fig.canvas.flush_events()
#pl.pause(2)

print("\n\n\n\n")
old_teams = teams
res_dict = {}
for n,i in enumerate(season.T):
    id_h = np.where(teams[0] == i[2])
    id_a = np.where(teams[0] == i[3])

    elo_h = teams[1][id_h]
    elo_a = teams[1][id_a]

    prob = p(elo_h,elo_a)

    if i[4] > i[5]:
        point = 1
    elif i[4] < i[5]:
        point = 0
    else:
        point = 0.5

    f_delta = point - prob

    if np.abs(f_delta) > 0.8:
        pass
        #print(i)

    lam_h = poly_3(prob,*popt_poly)[0]
    lam_a = poly_3(1-prob,*popt_poly)[0]

    s_h,s_a = prob_poiss(lam_h,lam_a,1)

    p_h = 3 if s_h > s_a else 1 if s_h == s_a else 0
    p_a = 3 if s_h < s_a else 1 if s_h == s_a else 0

    try:
        res_dict[teams[0][id_h][0]] += p_h
        res_dict[teams[0][id_a][0]] += p_a
    except KeyError:
        res_dict[teams[0][id_h][0]] = p_h
        res_dict[teams[0][id_a][0]] = p_a

    #print(f"Pred: {s_h}:{s_a} --> True: {i[4]}:{i[5]}")
    """
    w_goals = i[4] if i[4] > i[5] else i[5]
    l_goals = i[4] if i[4] < i[5] else i[5]
    """
    w_goals = s_h if s_h > s_a else s_a
    l_goals = s_h if s_h > s_a else s_a
    margin = margin_victory_multiplier(w_goals,l_goals)

    change = k_factor*f_delta*margin

    teams[1][id_h] += change
    teams[1][id_a] += change

    line.remove()
    line = ax.barh(y_pos, teams[1], align='center',
                   color='green', ecolor='black')
    fig.suptitle(f"{i[0]}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    #pl.pause(0.1)
print("\n")
#print(sorted(res_dict,key=res_dict.get))
for w in sorted(res_dict, key=res_dict.get, reverse=True):
  print(w, res_dict[w])
pl.show()