import datetime
from microgrid.environments.industrial.industrial_env import IndustrialEnv


import datetime

import pulp

from microgrid.environments.industrial.industrial_env import IndustrialEnv

from pulp import *
import numpy as np

class IndustrialAgent:
    def __init__(self, env: IndustrialEnv):
        self.env = env

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):

        conso_prevision= state.get("consumption_prevision")  # consomation prévue
        manager_prix = state.get("manager_signal")  # les prix

        pmax = self.env.battery.pmax
        efficacite_batterie = self.env.battery.efficiency
        capacite_batterie = self.env.battery.capacity

        T = [t for t in range(24)]
        stock_init = state.battery.initial_soc  # chargement initial de la batterie
        stock = pulp.LpVariable.dicts("batterie_stock", T, 0, self.env.battery.capacity)



        pb = pulp.LpProblem("industrial_site", pulp.LpMinimize)


        l_batterie_plus = pulp.LpVariable.dicts("l_batterie_plus", T, 0)
        l_batterie_moins = pulp.LpVariable.dicts("l_batterie_moins", T, 0)
        l_batterie = pulp.LpVariable.dicts("l_batterie", T)
        l_tot = pulp.LpVariable.dicts("l_demande_totale", T)




#contraintes
        pb += stock[0] == stock_init, "égalité des stocks au temps 0"

        for t in range(24):
            pb += l_batterie[t] == -l_batterie_moins[t] + l_batterie_plus[t] , "égalité des utilisations de la batterie"
            pb += l_batterie_moins[t] + l_batterie_plus[t] <= pmax ,"maximisation de l'utilisattion de la batterie"
            pb += l_tot[t] == l_batterie[t] + conso_prevision[t] , "égalité entre la demande et l'arrivée d'énergie"
            pb += stock[t] <= capacite_batterie, "maximisation de la capacité de la batterie"
            if t > 0 :

                pb += stock[t] == stock[t-1]  + delta_t*(l_batterie_plus[t]*efficacite_batterie + l_batterie_moins[t]/efficacite_batterie) , " égalité des stocks"

#fonction obj
        pb += pulp.lpSum([l_tot[t] * manager_prix[t] * delta_t for t in T])  # On somme les prix et on cherche à les minimiser

        pb.solve()
        decision = np.zeros(24)
        for i in range(24):
            decision[i] = l_tot[i]

        return decision


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    industrial_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'building': {
            'site': 1,
        }
    }
    env = IndustrialEnv(industrial_config=industrial_config, nb_pdt=N)
    agent = IndustrialAgent(env)
    cumulative_reward = 0
    now = datetime.datetime.now()
    state = env.reset(now, delta_t)
    for i in range(N*2):
        action = agent.take_decision(state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))
