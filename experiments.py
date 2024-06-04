import os
import sys

from tqdm.contrib.concurrent import process_map
import numpy as np

from agent import UCRLAdmissionControl
from system import System
from utils import PolicyIteration

NB_EXPERIMENTS = 100
NB_CPU = os.cpu_count()


class ExpUCRLAC:
    def __init__(
        self,
        paramereters,
        time_max=int(1e6),
        first_episode_length=10,
        lambda_min=0.1,
        lambda_max=1,
        name="exp",
        optimal_avg_reward=0,
    ) -> None:
        self.parameters = paramereters
        mu_max = (
            min(paramereters["servers"], paramereters["size"])
            * paramereters["service rate"]
        )
        if first_episode_length is None:
            self.first_episode_length = 1 / mu_max + 1e-6
        else:
            self.first_episode_length = first_episode_length
        self.time_max = time_max
        self.name = name
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.optimal_avg_reward = optimal_avg_reward

    def run(self, seed=0):
        self.seed = seed
        exp_path = os.path.join(
            os.getcwd(), "experiments", self.name, f"exp_{seed:03d}"
        )
        os.makedirs(exp_path)
        self.exp_path = exp_path

        agent = UCRLAdmissionControl(
            parameters=self.parameters,
            first_episode_length=self.first_episode_length,
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max,
            exp_path=self.exp_path,
        )
        queueing_system = System(parameters=self.parameters, initial_seed=self.seed)
        total_reward = 0
        regret = []
        times = []
        i = 0

        state, arriving_job_class, info = queueing_system.reset()

        while info["time"] < self.time_max:
            action = agent.get_action(state, arriving_job_class, info)
            new_state, new_arriving_job_class, reward, new_info = queueing_system.step(
                action
            )
            total_reward += reward
            if i % 1000 == 0:
                regret.append(self.optimal_avg_reward * info["time"] - total_reward)
                times.append(info["time"])
            i += 1
            state, arriving_job_class, info = (
                new_state,
                new_arriving_job_class,
                new_info,
            )
        self.save(regret, times)

    def save(self, regret, times):
        np.save(os.path.join(self.exp_path, "regret.npy"), np.array(regret))
        np.save(os.path.join(self.exp_path, "times.npy"), np.array(times))


if __name__ == "__main__":
    service_rate = float(sys.argv[1])
    size = int(sys.argv[2])
    total_time = int(1e6)
    nb_experiments = NB_EXPERIMENTS
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

    holding_cost = np.array(
        [
            [
                coefs[i]
                * (s - n_servers + 1)
                / (n_servers * service_rate)
                * (s >= n_servers)
                for i in range(nb_classes)
            ]
            for s in range(size + 1)
        ]
    )

    parameters = {
        "servers": n_servers,
        "classes": nb_classes,
        "size": size,
        "service rate": service_rate,
        "arrival rates": arrival_rate[:nb_classes],
        "rewards": rewards,
        "holding costs": holding_cost,
    }
    lambda_min = 1
    lambda_max = 4
    algoPI = PolicyIteration(parameters)
    tmp_policy = np.ones((size + 1, nb_classes), dtype=int)
    optimal_policy, optimal_avg_reward, _ = algoPI.PI(tmp_policy)

    print(f"Optimal average reward: {optimal_avg_reward.item()}")
    algoPI = PolicyIteration(parameters)
    tmp_policy = np.ones((parameters["size"] + 1, parameters["classes"]), dtype=int)
    _, optimal_avg_reward, _ = algoPI.PI(tmp_policy)

    Exp = ExpUCRLAC(
        parameters,
        time_max=total_time,
        name=os.path.join("UCRL-AC", f"S{size}_mu{service_rate}"),
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        optimal_avg_reward=optimal_avg_reward,
    )

    process_map(Exp.run, range(NB_EXPERIMENTS), max_workers=4)
