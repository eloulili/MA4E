import datetime
from microgrid.environments.data_center.data_center_env import DataCenterEnv


class DataCenterAgent:
    def __init__(self, env: DataCenterEnv):
        self.env = env


    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
            l_IT = state['consumption_prevision']  # on étudie le scénario à partir de la consommation minimale entre t et t+24h
            lambdas = state["manager_signal"]
            phw = state["hotwater_price_prevision"]
            alpha = [0 for i in range(self.env.nb_pdt)]
            l_NF = [0 for i in range(self.env.nb_pdt)]
            h_r = [0 for i in range(self.env.nb_pdt)]

            for t in range(self.env.nb_pdt):
                l_NF[t] = (1 + 1 / (self.env.EER * delta_t)) * l_IT[t]
                h_r[t] = l_IT[t] * self.env.COP_CS / self.env.EER
                max_alpha = self.env.max_transfert*(self.env.COP_HP -1)/(self.env.COP_HP * h_r[t])
                if phw[t]>= lambdas[t]*datetime.timedelta(hours=1)/ (self.env.COP_HP * self.env.delta_t) and l_IT[t] > 0:
                    if max_alpha>=0 :
                        alpha[t] = min(max_alpha,1)
                    else :
                        alpha[t] = 0
            return alpha




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



