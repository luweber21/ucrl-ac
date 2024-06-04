from collections import deque

import numpy as np
from numpy.random import default_rng


def is_empty(queue: deque) -> bool:
    return bool(queue) == False


class LeavingJobs:
    def __init__(self, parameters, seed=2) -> None:
        self._seed: int = seed
        self._n_servers = parameters["servers"]
        self._service_rate = parameters["service rate"]
        self._queue_size = parameters["size"]
        self.reset()

    def reset(self) -> None:
        self._rng = default_rng(self._seed)
        self._queues = [deque() for _ in range(self._n_servers)]
        [self._fill_the_queue(n) for n in range(self._n_servers)]

    def _new_departures(self, n) -> np.ndarray:
        return self._rng.exponential(
            scale=1 / ((n + 1) * self._service_rate), size=int(1e6)
        )

    def _fill_the_queue(self, n) -> None:
        [self._queues[n].append(departure) for departure in self._new_departures(n)]

    def get_next_departure(self, n) -> float:
        if n == 0:
            return np.infty
        n = min(n, self._n_servers)
        if is_empty(self._queues[n - 1]):
            self._fill_the_queue(n - 1)
        return self._queues[n - 1].popleft()


class IncomingJobs:
    """
    Class representing the future jobs.
    It contains a queue of tuples representing the inter-arrival times and the job classes.
    This class is used to decrease the calls to random generators.
    """

    def __init__(self, parameters, seed=1) -> None:
        self._seed: int = seed
        self._n_classes = parameters["classes"]
        self._arrival_rates = parameters["arrival rates"]
        self._global_arrival_rate: float = self._arrival_rates.sum()
        self.reset()

    def _new_arrivals(self) -> np.ndarray:
        return self._rng.exponential(scale=1 / self._global_arrival_rate, size=int(1e6))

    def _new_types(self) -> np.ndarray:
        return self._rng.choice(
            range(self._n_classes),
            size=int(1e6),
            p=self._arrival_rates / self._global_arrival_rate,
        )

    def _fill_the_queue(self) -> None:
        [
            self._queue.append((x, customer_type))
            for (x, customer_type) in zip(self._new_arrivals(), self._new_types())
        ]

    def get_next_job(self) -> tuple[float, int]:
        if is_empty(self._queue):
            self._fill_the_queue()
        return self._queue.popleft()

    def reset(self) -> None:
        self._rng = default_rng(self._seed)
        self._queue = deque()
        self._fill_the_queue()

    def test(self, expected_global_arrival_rate) -> None:
        np.testing.assert_almost_equal(
            np.mean(self._new_arrivals()), 1 / expected_global_arrival_rate, 3
        )
        print(
            f"Empirical inter-arrival times correspond to the global arrival rate (={expected_global_arrival_rate})."
        )
