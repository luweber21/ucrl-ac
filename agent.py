import abc

import numpy as np

from utils import discrete_matshow


class AgentInterface(metaclass=abc.ABCMeta):
    def __init__(self, parameters):
        self.parameters = parameters

    @abc.abstractmethod
    def get_action(self, state: int, job_class: int, info: dict) -> bool:
        pass


class RandomAgent(AgentInterface):
    def __init__(self, parameters):
        super().__init__(parameters)

    def get_action(self, state, job_class, info) -> bool:
        if state < self.parameters["size"]:
            return np.random.randint(2, dtype=bool)
        return False


class FixedPolicyAgent(AgentInterface):
    def __init__(self, parameters, policy):
        super().__init__(parameters)
        self.policy = policy

    def get_action(self, state, job_class, info) -> bool:
        if state < self.parameters["size"]:
            return self.policy[state, job_class]
        else:
            return False


class UCRLAdmissionControl(AgentInterface):
    def __init__(
        self, parameters, first_episode_length, lambda_min, lambda_max, exp_path
    ):
        super().__init__(parameters)
        queue_size = self.parameters["size"]
        n_classes = self.parameters["classes"]
        self.Tk = first_episode_length
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.mu_max = (
            min(parameters["servers"], parameters["size"]) * parameters["service rate"]
        )

        self.optimistic_global_arrival_rate = self.lambda_max
        self.optimistic_classes_distributions = np.array(
            [[1] + [0] * (n_classes - 1)] * (queue_size + 1)
        ).reshape(queue_size + 1, n_classes)
        self.job_distribution_estimators = np.zeros(n_classes)
        self.inverse_of_global_arrival_rate_estimator = 0
        self.last_arrival = 0
        self.nb_arrivals = np.zeros(n_classes, dtype=int)
        self.new_arrivals = 0
        self.exp_path = exp_path

        self.episode = 1
        self.policy = get_initial_policy(
            parameters,
            self.optimistic_global_arrival_rate,
            self.optimistic_classes_distributions,
        )

        self.expected_rewards = np.array(
            [
                self.parameters["rewards"][i] - self.parameters["holding costs"][s, i]
                for s in range(queue_size + 1)
                for i in range(n_classes)
            ]
        ).reshape(queue_size + 1, n_classes)

        self.class_priorities = self.expected_rewards.copy()
        for state in range(queue_size + 1):
            self.class_priorities[state] = np.argsort(self.class_priorities[state])[
                ::-1
            ]
        self.class_priorities = self.class_priorities.astype(int)
        # discrete_matshow(self.parameters, self.policy, self.exp_path)

    def update_policy(self) -> None:
        epsilon_tau = (
            4
            / self.lambda_min
            * np.sqrt(2 * np.log(self.mu_max * self.Tk) / self.new_arrivals)
        )

        if epsilon_tau < self.inverse_of_global_arrival_rate_estimator:
            self.optimistic_global_arrival_rate = min(
                1 / (self.inverse_of_global_arrival_rate_estimator - epsilon_tau),
                self.lambda_max,
                1 / self.inverse_of_global_arrival_rate_estimator
                + self.lambda_max**2 * epsilon_tau,
            )
        else:
            self.optimistic_global_arrival_rate = min(
                1 / self.inverse_of_global_arrival_rate_estimator
                + self.lambda_max**2 * epsilon_tau,
                self.lambda_max,
            )

        self.new_arrivals = 0
        self.inv_estimator = 0
        self.Tk *= 2

        self.get_optimistic_distribution()
        self.policy = policy_iteration(
            self.policy,
            self.parameters,
            self.optimistic_global_arrival_rate,
            self.optimistic_classes_distributions,
        )["policy"]
        self.episode += 1
        # discrete_matshow(self.parameters, self.policy, self.exp_path)

    def get_optimistic_distribution(self):
        job_distribution = self.nb_arrivals / self.nb_arrivals.sum()
        S = self.parameters["size"]
        m = self.parameters["classes"]
        delta = self.mu_max * self.Tk
        N = self.nb_arrivals.sum()
        w = np.sqrt(2 * m * np.log(2 * delta) / N)
        prio = self.class_priorities

        optimistic_distributions = np.zeros((S + 1, m))
        for s in range(S + 1):
            d = w / 2
            optimistic_distributions[s] = job_distribution
            for i in range(m):
                if 1 - job_distribution[prio[s][i]] >= d:
                    optimistic_distributions[s][prio[s][i]] = (
                        job_distribution[prio[s][i]] + d
                    )
                    break
                else:
                    d -= 1 - job_distribution[prio[s][i]]
                    optimistic_distributions[s][prio[s][i]] = 1
            i = m - 1
            while optimistic_distributions[s].sum() > 1:
                optimistic_distributions[s][prio[s][i]] = max(
                    0,
                    1
                    - optimistic_distributions[s].sum()
                    + optimistic_distributions[s][prio[s][i]],
                )
                i = i - 1

        self.optimistic_classes_distributions = optimistic_distributions

    def get_action(self, state, job_class, info) -> bool:
        if info["time"] > self.Tk:
            self.update_policy()

        inter_arrival_time = info["time"] - self.last_arrival
        self.nb_arrivals[job_class] += 1
        self.new_arrivals += 1
        self.last_arrival = info["time"]
        self.inverse_of_global_arrival_rate_estimator += (
            1
            / self.new_arrivals
            * (
                inter_arrival_time
                * (
                    inter_arrival_time
                    <= (2 * self.new_arrivals)
                    / (self.lambda_min**2 * np.log(self.mu_max * self.Tk))
                )
                - self.inverse_of_global_arrival_rate_estimator
            )
        )

        if state < self.parameters["size"]:
            return self.policy[state, job_class]
        else:
            return False


def get_initial_policy(
    parameters, optimistic_global_arrival_rate, optimistic_classes_distributions
) -> np.ndarray:
    queue_size = parameters["size"]
    n_classes = parameters["classes"]
    initial_policy = np.zeros((queue_size + 1, n_classes), dtype=int)
    initial_policy[:, 0] = 1
    return policy_iteration(
        initial_policy,
        parameters,
        optimistic_global_arrival_rate,
        optimistic_classes_distributions,
    )["policy"]


def improvement(relative_bias, parameters) -> np.ndarray:
    new_pi = np.zeros((parameters["size"] + 1, parameters["classes"]), dtype=int)
    for state in range(len(relative_bias)):
        for job_class in range(parameters["classes"]):
            new_pi[state, job_class] = (
                parameters["rewards"][job_class]
                - parameters["holding costs"][state, job_class]
                >= relative_bias[state]
            )
    return new_pi


def evaluation(
    policy, parameters, optimistic_global_arrival_rate, optimistic_classes_distributions
) -> tuple[float, np.ndarray]:
    """
    Returns the average reward and the relative bias h(s)-h(s+1).
    """
    queue_size = parameters["size"]
    service_rate = parameters["service rate"]
    n_servers = parameters["servers"]
    n_classes = parameters["classes"]
    rewards = parameters["rewards"]
    holding_cost = parameters["holding costs"]

    M = np.zeros((queue_size + 1, queue_size + 1))
    for state in range(queue_size + 1):
        M[state, state] = -service_rate * min(n_servers, state)
        M[state, 0] = 1
        if state < queue_size:
            M[state, state + 1] = optimistic_global_arrival_rate * sum(
                optimistic_classes_distributions[state] * policy[state]
            )

    R = np.zeros((queue_size + 1, 1))
    for state in range(queue_size + 1):
        reward = 0
        for job_class in range(n_classes):
            if policy[state, job_class]:
                reward += (
                    rewards[job_class] - holding_cost[state, job_class]
                ) * optimistic_classes_distributions[state][job_class]
        R[state, 0] = optimistic_global_arrival_rate * reward

    relative_bias = np.linalg.solve(M, R)
    return relative_bias[0], relative_bias[1:]


def policy_iteration(
    policy, parameters, optimistic_global_arrival_rate, optimistic_classes_distributions
) -> dict:
    loop = True
    n = 0
    while loop:
        average_reward, relative_bias = evaluation(
            policy,
            parameters,
            optimistic_global_arrival_rate,
            optimistic_classes_distributions,
        )
        new_policy = improvement(relative_bias, parameters)
        n += 1
        loop = (not np.array_equal(new_policy, policy)) and (n < 1000)
        policy = new_policy
    return {
        "policy": policy,
        "average reward": average_reward,
        "relative bias": relative_bias,
    }
