import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class PolicyIteration:
    def __init__(self, parameters):
        self.parameters = parameters

    def evaluation(self, pi):
        S = self.parameters["size"] + 1
        c = self.parameters["servers"]
        lamdas = self.parameters["arrival rates"]
        lamda = sum(lamdas)
        lamdas = lamdas / lamda

        M = np.zeros((S, S))
        for s in range(S):
            M[s, s] = -self.parameters["service rate"] * min(c, s)
            M[s, 0] = 1
            if s < S - 1:
                M[s, s + 1] = lamda * sum(lamdas * pi[s])

        R = np.zeros((S, 1))
        for s in range(S):
            rew = 0
            for i in range(self.parameters["classes"]):
                if pi[s, i]:
                    rew += (
                        self.parameters["rewards"][i]
                        - self.parameters["holding costs"][s, i]
                    ) * lamdas[i]
            R[s, 0] = lamda * rew

        relative_bias = np.linalg.solve(M, R)
        return relative_bias[0], relative_bias[1:]

    def improvement(self, h):
        new_pi = np.zeros(
            (self.parameters["size"] + 1, self.parameters["classes"]), dtype=int
        )
        for s in range(len(h)):
            for i in range(self.parameters["classes"]):
                new_pi[s, i] = (
                    self.parameters["rewards"][i]
                    - self.parameters["holding costs"][s, i]
                    >= h[s]
                )
        return new_pi

    def PI(self, pi):
        loop = True
        n = 0
        while loop:
            avg_reward, bias = self.evaluation(pi)
            new_pi = self.improvement(bias)
            n += 1
            loop = (not np.array_equal(new_pi, pi)) and (n < 1000)
            pi = new_pi
        if n >= 100:
            print("PI has not converged.")
        return pi, avg_reward, bias


def discrete_matshow(parameters, policy, exp_path):
    PIAlgo = PolicyIteration(parameters)
    optimal_policy, optimal_avg_reward, _ = PIAlgo.PI(policy)

    X, Y = np.meshgrid(
        np.arange(parameters["classes"]), np.arange(parameters["size"] + 1)
    )
    z = np.zeros(policy.shape)
    for s in range(z.shape[0]):
        for i in range(z.shape[1]):
            if policy[s, i] == 1 and optimal_policy[s, i] == 1:
                z[s, i] = -1
            elif policy[s, i] == 0 and optimal_policy[s, i] == 0:
                z[s, i] = 0
            elif policy[s, i] == 1 and optimal_policy[s, i] == 0:
                z[s, i] = 1
            else:
                z[s, i] = 2

    plt.figure()
    cmap = matplotlib.colors.ListedColormap(["green", "gray", "blue", "red"])
    boundaries = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    plt.pcolor(X, Y, z, edgecolors="k", linewidths=1, snap=True, cmap=cmap, norm=norm)
    labels = [
        "current policy: 0 / optimal policy: 1",
        "current policy: 0 / optimal policy: 0",
        "current policy: 1 / optimal policy: 0",
        "current policy: 0 / optimal policy: 1",
    ]
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
    plt.colorbar(ticks=np.arange(np.min(-1), np.max(2) + 1), format=fmt)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.title(
        f"optimal: {float(optimal_avg_reward):.2f} \ncurrent: {float(PIAlgo.evaluation(policy)[0]):.2f}"
    )
    img_dir = os.path.join(exp_path, "img")
    try:
        os.makedirs(img_dir)
    except:
        pass
    plt.savefig(os.path.join(img_dir, f"episode_{len(os.listdir(img_dir)) + 1}.png"))
    plt.close()
