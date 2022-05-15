import datetime
from microgrid.environments.data_center.data_center_env import DataCenterEnv
from microgrid.assets.data_center import DataCenter

import pandas as pd
from pulp import *
import numpy as np

class DataCenterAgent:
    def __init__(self, env: DataCenterEnv, Dt : DataCenter):
        self.env = env
        self.Dt = Dt

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
            l_IT = np.array([self.Dt.data[i+ self.env.now]] for i in range(
                self.env.nb_pdt))  # on étudie le scénario à partir de la consommation minimale entre t et t+24h
            lambdas = self.env.observation_space["manager_signal"]
            phw = self.Dt.prices
            problem = LpProblem("data_center", LpMinimize)
            alphas = [0 for i in range(self.env.nb_pdt)]
            alpha = [0 for i in range(self.env.nb_pdt)]
            for i in range(24):
                var_name = "alpha_" + str(i)
                alphas[i] = LpVariable(var_name, 0.0, 1.0)

            l_NF = [0 for i in range(self.env.nb_pdt)]
            h_r = [0 for i in range(self.env.nb_pdt)]
            l_HP = [0 for i in range(self.env.nb_pdt)]
            h_DC = [0 for i in range(self.env.nb_pdt)]
            li = [0 for i in range(self.env.nb_pdt)]

            for t in range(self.env.nb_pdt):
                l_NF[t] = (1 + 1 / (self.env.EER * self.env.delta_t)) * l_IT[t]
                h_r[t] = l_IT[t] * self.env.COP_CS / self.env.EER
                l_HP[t] = alphas[t] * h_r[t] / ((self.env.COP_HP - 1) * dt)
                h_DC[t] = self.env.COP_HP * self.env.delta_t * l_HP[t]
                cons_name = "production limite en " + str(t)
                problem += h_DC[t] <= self.env.max_transfert
                li[t] = l_HP[t] + l_NF[t]

            problem += np.sum([lambdas[i] * (l_NF[i] + h_DC[i]) - phw[i] * h_DC[i] for i in
                               range(self.env.nb_pdt)]), "objectif"

            problem.solve()
            for i in range(48):
                alpha[i] = alphas[i].value()
            self.env.action_space = alpha
            return self.env.action_space.sample()


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    data_center_config = {
        'scenario': 10,
    }
    env = DataCenterEnv(data_center_config, nb_pdt=N)
    agent = DataCenterAgent(env)
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