from hiive import mdptoolbox
import numpy as np
from graphing import graph
import time



def pi_taxi(gamma=0.9):
    np.random.seed(4171991)
    P, R = mdptoolbox.example.openai("Taxi-v3")
    pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1)
    start = time.time()
    pi.run()
    end = time.time() - start
    data = np.zeros((len(pi.run_stats), 4))
    stats = pi.run_stats
    for i, stat in enumerate(stats):
        data[i, 0] = stat['Iteration']
        data[i, 1] = stat['Max V']
        data[i, 2] = stat['Error']
        data[i, 3] = stat['Reward']
    graph(data, "PI Taxi", filename="PI/Taxi")
    print(pi.policy)
    print(data[-1, 3], end)


def pi_forest(S=4, r1=60, r2=50, gamma=0.9):
    np.random.seed(4171991)
    P, R = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=0.1)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
    start = time.time()
    pi.run()
    end = time.time() - start
    data = np.zeros((len(pi.run_stats), 4))
    stats = pi.run_stats
    for i, stat in enumerate(stats):
        data[i, 0] = stat['Iteration']
        data[i, 1] = stat['Max V']
        data[i, 2] = stat['Error']
        data[i, 3] = stat['Reward']
    graph(data, "PI Forest", filename="PI/Forest")
    print(pi.policy)
    print(data[-1, 3], end)




