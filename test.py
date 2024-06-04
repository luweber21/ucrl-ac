import os
import sys

import numpy as np
from tqdm import tqdm

from agent import RandomAgent
from system import System


def run_test(parameters, seed, time_max):
    agent = RandomAgent(
        parameters=parameters,
    )
    queueing_system = System(parameters=parameters, initial_seed=seed)
    queueing_system.test(parameters["arrival rates"].sum())

    total_reward = 0
    arrival_times = []

    state, arriving_job_class, info = queueing_system.reset()
    arrival_times.append(info["time"])

    while info["time"] < time_max:
        action = agent.get_action(state, arriving_job_class, info)
        new_state, new_arriving_job_class, reward, new_info = queueing_system.step(
            action
        )
        arrival_times.append(new_info["time"])
        total_reward += reward
        state, arriving_job_class, info = (
            new_state,
            new_arriving_job_class,
            new_info,
        )
    arrival_times = np.array(arrival_times)
    inter_arrival_times = arrival_times[1:] - arrival_times[:-1]
    np.testing.assert_almost_equal(
        np.mean(inter_arrival_times.mean()), 1 / parameters["arrival rates"].sum(), 3
    )
    print(
        f"Empirical inter-arrival times correspond to the global arrival rate (={parameters['arrival rates'].sum()})."
    )


if __name__ == "__main__":
    service_rate = float(sys.argv[1])
    size = int(sys.argv[2])
    total_time = int(1e6)
    nb_classes = 2

    coefs = np.array([0.1, 0.1])
    arrival_rate = np.array(
        [
            1,
            1,
        ]
    )

    rewards = np.array([20, 10])

    n_servers = 5

    parameters = {
        "servers": n_servers,
        "classes": nb_classes,
        "size": size,
        "service rate": service_rate,
        "arrival rates": arrival_rate[:nb_classes],
        "rewards": rewards,
        "holding costs": lambda s, i: coefs[i]
        * (s - n_servers + 1)
        / (n_servers * service_rate)
        * (s >= n_servers),
    }
    seed = 0

    run_test(parameters, seed, total_time)
