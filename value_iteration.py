from hiive import mdptoolbox
import numpy as np
from graphing import graph
import time


def vi_taxi(gamma=0.9):
    np.random.seed(4171991)
    P, R = mdptoolbox.example.openai("Taxi-v3")
    pi = mdptoolbox.mdp.ValueIteration(P, R, gamma)
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
    graph(data, "VI Taxi", filename="VI/Taxi")
    print(pi.policy)
    print(data[-1, 3], end)



def vi_forest(gamma=0.9):
    np.random.seed(4171991)
    P, R = mdptoolbox.example.forest(S=4, r1=60, r2=50, p=0.1)
    pi = mdptoolbox.mdp.ValueIteration(P, R, gamma)
    start = time.time()
    pi.run()
    end = time.time() - start
    data = np.zeros((len(pi.run_stats), 4))
    stats = pi.run_stats
    print(stats)
    for i, stat in enumerate(stats):
        data[i, 0] = stat['Iteration']
        data[i, 1] = stat['Max V']
        data[i, 2] = stat['Error']
        data[i, 3] = stat['Reward']
    graph(data, "VI Forest", filename="VI/Forest")
    print(pi.policy)
    print(data[-1, 3], end)


