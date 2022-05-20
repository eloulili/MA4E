import datetime
from microgrid.environments.charging_station.charging_station_env import ChargingStationEnv


class ChargingStationAgent:
    def __init__(self, env: ChargingStationEnv):
        self.env = env

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
        #Creation du problème linéaire
        lp = pulp.LpProblem("charging_station"+".lp", pulp.LpMinimize)
        lp.setSolver()
        l = {} 
        a = {}
        fined = {}
        T = self.env.nb_pdt + 1 #Nombre de périodes temporelles
        J = self.env.nb_evs + 1 #Nombre de véhicules
        dt = self.env.delta_t/datetime.timedelta(hours=1)
        #Variables fixes car je ne les ai pas trouvées dans l'environnement
        daily_conso = 4. 
        fine = 5. #Lorsque la batterie est inférieure à 25%

        #On récupère dans l'environnement les temps de départ et d'arrivés des véhicules électriques, que l'on met sous forme de tuple dans t_dep_arr
        #La liste t_ini permet de récupérer le temps auquel la voiture est branchée pour la première fois
        t_dep_arr = []
        t_ini = []    
        for j in range(1,J):
            dep = []
            t = 0
            while t <= T - 2:  
                k = 0
                while t + k < T - 1 and state["is_plugged_prevision"][j-1][t + k] == state["is_plugged_prevision"][j-1][t]:
                    k += 1
                if state["is_plugged_prevision"][j-1][t] == 0 and t!=0: #Si t==0, alors cela veut dire que la voiture n'était pas branchée, donc on ne se trouve pas dans une situation de départ
                    dep.append([t,t+k])
                t += k + 1
            if dep!= [] and dep[-1][-1] == T-1:
                dep[-1][-1] = None
            t_dep_arr.append(dep)
            t_init = 1
            while t_init <= T - 1 and state["is_plugged_prevision"][j-1][t_init-1] == 0:
                t_init += 1
            t_ini.append(t_init)

        #Récupération dans l'environnement du signal du manageur et de l'état initial de chaque voiture
        lbd = state["manager_signal"]
        a_ini = state["soc"]

        #Création des variables
        for j in range(1,J):
            l[j] = {} 
            a[j] = {}
            fined[j] = {}
            for (i,el) in enumerate(t_dep_arr[j-1]):
                var_name = "fined_"+str(j)+'_'+str(i)
                fined[j][i] = pulp.LpVariable(var_name,cat="Binary")
            for t in range(1,T):
                var_name = "l_"+str(j)+"_"+str(t)
                l[j][t] = pulp.LpVariable(var_name, self.env.evs[j-1].battery.pmin, self.env.evs[j-1].battery.pmax)
                var_name = "a_" + str(j)+"_"+str(t)
                a[j][t] = pulp.LpVariable(var_name,0.,self.env.evs[j-1].battery.capacity)
        #l[j][t] est la puissance dédiée à la voiture j au temps t
        # a[j][t] est l'état de la batterie de la voiture j au temps t
        # fined[j][i] est un booléen qui nous indique si la voiture j part avec une batterie chargée à moins de 25% de sa capacité lors du pas de temps t_dep[i]    


        #Ajout des contraintes:
        for t in range(1,T):
            #On impose à chaque pas de temps que la somme des puissances données ou récupérées 
            # des batteries soit en module inférieure à la capacité de la station.
            const_name = "charging_station_capacity_positive_"+str(t)
            lp += pulp.lpSum([ l[j][t] for j in range(1,J)]) <= self.env.pmax_site, const_name
            const_name = "charging_station_capacity_respected_negative_"+str(t)
            lp += pulp.lpSum([ l[j][t] for j in range(1,J)]) >= - self.env.pmax_site, const_name
        for j in range(1,J):
            if t_dep_arr[j-1] != []: #S'il y a un départ pour la voiture j
                for (i,el) in enumerate(t_dep_arr[j-1]):
                    const_name = "recharged_at_least_at_10_percent_ev_"+str(j)+"_"+str(i) #Comme on veut minimiser une fonction croissante en fined, cette inégalité suffit pour imposer 
                    #fined[j][i] == (a[j][el[0]] <= self.env.evs[j-1].battery.capacity*0.25)
                    lp += (a[j][el[0]]>=self.env.evs[j-1].battery.capacity*0.25*(1-fined[j][i])),const_name
                    #On vérifie qu'au départ, la voiture j a assez d'énergie pour rouler
                    const_name = "required_energy_"+str(j)+'_'+str(i)
                    lp += a[j][el[0]] >= daily_conso,const_name
                    #On impose des relations entre les state of charge:
                    if el[-1] != None:
                        const_name = "daily_conso_"+str(j)+"_"+str(el[1])+"_"+str(i)
                        lp += (a[j][el[1]] == a[j][el[0]] - daily_conso),const_name
                        #On impose qu'à l'arrivée on est la bonne relation entre les états de la batterie
                        const_name = "evolution_state_"+str(j)+'_'+str(el[1])
                        lp += (a[j][el[1]+1] == a[j][el[1]] + self.env.evs[j-1].battery.efficiency*l[j][el[1]]*dt), const_name
            #On impose un état de charge initial à la première arrivée à la borne
            const_name = "initial_state_ev_"+str(j)
            lp += (a[j][t_ini[j-1]] == a_ini[j-1]), const_name
            

            for t in range(1,T):
                #Si la voiture n'est pas en charge, alors on impose que la puissance échangée soit nulle
                if not (state["is_plugged_prevision"][j-1][t-1]):
                    const_name = "away_from_charging_station_"+str(j)+"_"+str(t)
                    lp += l[j][t]==0.,const_name
                #Sinon, on modifie l'état de la batterie en fonction de la puissance délivrée
                else:
                    if t < T-1:
                        const_name = "evolution_state_"+str(j)+'_'+str(t)
                        lp += (a[j][t+1] == a[j][t] + self.env.evs[j-1].battery.efficiency*l[j][t]*0.5), const_name
            
        #On cherche à minimiser le coût, qui correspond au prix de l'électricité * la puissance échangée (achetée ou vendue) à chaque pas de temps
        #Plus les éventuelles amendes liées au départ des voitures
        lp.setObjective(pulp.lpSum(pulp.lpSum(l[j][t] * lbd[t-1] for j in range(1,J)) for t in range(1,T)) + pulp.lpSum(pulp.lpSum(fined[j][i] for i in range(len(t_dep_arr[j-1])))for j in range(1,J)) * fine)
 
        #On résout et on renvoie notre action. 
        lp.solve()
        res = self.env.action_space.sample()
        for j in range(0,J-1):
            for t in range(0,T-1):
                res[j][t] = l[j+1][t+1].varValue
        #print(res)
        return(res) 


    
if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    evs_config = [
        {
            'capacity': 40,
            'pmax': 22,
        },
        {
            'capacity': 40,
            'pmax': 22,
        },
        {
            'capacity': 40,
            'pmax': 3,
        },
        {
            'capacity': 40,
            'pmax': 3,
        },
    ]
    station_config = {
        'pmax': 40,
        'evs': evs_config
    }
    env = ChargingStationEnv(station_config=station_config, nb_pdt=N)
    agent = ChargingStationAgent(env)
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
        print("Info: {}".format(action.sum(axis=0)))
