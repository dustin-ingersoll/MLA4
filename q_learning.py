from hiive import mdptoolbox
import numpy as np
from graphing import graph
import time

np.random.seed(4171991)


def Q_taxi(epsilon=0.001):
    P, R = mdptoolbox.example.openai("Taxi-v3")
    pi = mdptoolbox.mdp.QLearning(P, R, 0.9, n_iter=50000)
    pi.run_stat_frequency = 1000
    np.random.seed(4171991)
    # pi.max_iter=1000
    pi.alpha_decay = 0.9999
    # pi.epsilon_decay = 0.999
    start = time.time()
    pi.run()
    end = time.time() - start
    data = np.zeros((len(pi.run_stats), 4))
    stats = pi.run_stats
    conv_iter = converge_error(pi, 0.00001)
    for i, stat in enumerate(stats):
        data[i, 0] = stat['Iteration']
        data[i, 1] = stat['Max V']
        data[i, 2] = stat['Error']
        data[i, 3] = abs(stats[i - 1]['Error'] - stat['Error'])
    graph(data, "Q Taxi", conv_iter, filename="Q/Taxi")
    print(pi.policy)
    print(data[-1, 1], end)



def Q_forest(epsilon=0.001):
    P, R = mdptoolbox.example.forest(S=4, r1=60, r2=50, p=0.1)
    # pi = mdptoolbox.mdp.QLearning(P, R, 0.9, alpha=0.9, alpha_min=0.001, alpha_decay=0.01)
    pi = mdptoolbox.mdp.QLearning(P, R, 0.9, n_iter=50000)
    pi.run_stat_frequency = 1000
    pi.alpha_decay = 0.9999
    np.random.seed(4171991)
    # pi.max_iter=1000
    start = time.time()
    pi.run()
    end = time.time() - start
    # stats = pi.run_stats
    conv_iter = converge_error(pi, 0.0001)

    # pi = mdptoolbox.mdp.QLearning(P, R, 0.9, n_iter=500000)
    # pi.run_stat_frequency = 10000
    # pi.alpha_decay = 0.999
    # pi.max_iter = conv_iter*1000
    # np.random.seed(4171991)
    # # pi.max_iter=1000
    # start = time.time()
    # pi.run()
    # end = time.time() - start
    stats = pi.run_stats
    data = np.zeros((len(pi.run_stats), 4))
    for i, stat in enumerate(stats):
        data[i, 0] = stat['Iteration']
        data[i, 1] = stat['Max V']
        data[i, 2] = stat['Error']
        data[i, 3] = abs(stats[i-1]['Error'] - stat['Error'])
    graph(data, "Q Forest", conv_iter, filename="Q/Forest")
    print(pi.policy)
    print(data[-1, 1], end)


def converge_error(pi, epsilon):
    i = 0
    converged = 0
    last = 0
    count = 0
    while not converged:
        run = pi.run_stats[i]
        this_error = run['Error']
        if abs(last - this_error) < epsilon:
            converged = i
            print(converged)
            print(pi.policy)
            break
        else:
            last = this_error
        i += 1
    return converged


def converge(run_stats):
    i = 0
    converged = 0
    last = 0
    count = 0
    while not converged:
        run = run_stats[i]
        this_v = run['Max V']
        buffer_high = last + (last * 0.0001)
        if this_v <= buffer_high:
            count += 1
            if count >= 10:
                converged = i
                print(converged)
                break
        else:
            last = this_v
            count = 0
        i += 1
    return converged

