from buffers import IncomingJobs, LeavingJobs


class System:
    def __init__(self, parameters, initial_seed=0):
        """
        System reproducing an Admission Control Problem in continuous time.
        """
        self._rewards = parameters["rewards"]
        self._hc = parameters["holding costs"]
        self._incoming_jobs = IncomingJobs(parameters, initial_seed)
        self._departures = LeavingJobs(parameters, initial_seed + 1)

    def step(self, job_is_accepted: bool) -> tuple[int, int, float, dict[str, float]]:
        if job_is_accepted:
            new_state = self._state + 1
            reward = (
                self._rewards[self._incoming_job_class]
                - self._hc[self._state, self._incoming_job_class]
            )
        else:
            new_state = self._state
            reward = 0
        incoming_arrival_time, incoming_job_class = self._incoming_jobs.get_next_job()
        departure_time = self._departures.get_next_departure(new_state)
        while departure_time < incoming_arrival_time:
            incoming_arrival_time -= departure_time
            new_state -= 1
            self._time += departure_time
            departure_time = self._departures.get_next_departure(new_state)

        self._incoming_job_class = incoming_job_class
        self._state = new_state
        self._time += incoming_arrival_time
        return self._state, self._incoming_job_class, reward, {"time": self._time}

    def reset(self) -> tuple[int, int, dict[str, float]]:
        self._state = 0
        self._incoming_jobs.reset()
        self._departures.reset()
        self._time, self._incoming_job_class = self._incoming_jobs.get_next_job()
        return self._state, self._incoming_job_class, {"time": self._time}

    def test(self, expected_global_arrival_rate):
        self._incoming_jobs.test(expected_global_arrival_rate)
