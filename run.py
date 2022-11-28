import value_iteration
import policy_iteration
import q_learning
import numpy as np


def run_taxi():
    print("===== VI TAXI =====")
    value_iteration.vi_taxi()
    print("===== PI TAXI =====")
    policy_iteration.pi_taxi()
    print("===== Q TAXI =====")
    q_learning.Q_taxi()


def run_forest():
    print("===== VI FOREST =====")
    value_iteration.vi_forest()
    print("===== PI FOREST =====")
    policy_iteration.pi_forest()
    print("===== Q FOREST =====")
    q_learning.Q_forest()


def run_VI():
    gamma_range = np.linspace(0.1, 0.9, 7)
    print("===== VI TAXI =====")
    for i in gamma_range:
        value_iteration.vi_taxi(i)
    print("===== VI FOREST =====")
    # for i in gamma_range:
    #     value_iteration.vi_forest(i)


def run_PI():
    print("===== PI TAXI =====")
    gamma_range = np.linspace(0.1, 0.9, 7)
    for i in gamma_range:
        policy_iteration.pi_taxi(i)
    print("===== PI FOREST =====")
    # for i in gamma_range:
    policy_iteration.pi_forest()


def run_Q():

    print("===== Q TAXI =====")
    q_learning.Q_taxi()
    print("===== Q FOREST =====")
    q_learning.Q_forest()


run_taxi()
run_forest()
# run_PI()
# run_VI()
# run_Q()