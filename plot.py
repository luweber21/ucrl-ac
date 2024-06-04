import numpy as np
import os
import pandas as pd


def get_regrets(nb_experiments, path):

    ucrlac_pi_regrets = []
    ucrlac_pi_times = []

    for n in range(nb_experiments):
        ucrlac_pi_regrets.append(
            np.load(os.path.join(path, f"exp_{n:03d}", "regret.npy"))
        )
        ucrlac_pi_times.append(np.load(os.path.join(path, f"exp_{n:03d}", "times.npy")))

    ucrlac_pi_support = np.concatenate(ucrlac_pi_times)
    ucrlac_pi_support.sort()

    for i in range(nb_experiments):
        res = np.interp(
            ucrlac_pi_support,
            ucrlac_pi_times[i].reshape(-1),
            ucrlac_pi_regrets[i].reshape(-1),
        )
        ucrlac_pi_regrets[i] = res

    ucrlac_pi_regrets = np.array(ucrlac_pi_regrets)

    return ucrlac_pi_regrets, ucrlac_pi_support


def get_curves(exp_path):
    nb_experiments = len(os.listdir(exp_path))
    regrets, support = get_regrets(nb_experiments, exp_path)
    mean = np.mean(regrets, axis=0)
    std = np.sqrt(np.var(regrets, axis=0) * nb_experiments / (nb_experiments - 1))
    lower_confidence_bound = mean - 1.96 * std / np.sqrt(nb_experiments)
    higher_confidence_bound = mean + 1.96 * std / np.sqrt(nb_experiments)
    return support, mean, lower_confidence_bound, higher_confidence_bound


def get_dataframe(exp_path, save_path, save=False):

    support, mean, lower_confidence_bound, higher_confidence_bound = get_curves(
        exp_path
    )
    res = pd.DataFrame()

    res.insert(0, "time", support)
    res.set_index("time", inplace=True)
    res.insert(0, "mean", mean)
    res.insert(1, "lower_conficende_bound", lower_confidence_bound)
    res.insert(2, "higher_confidence_bound", higher_confidence_bound)
    jumps = int(np.round(res.shape[0] / 1000))
    res = res.iloc[::jumps]
    print(res.head())

    if save:
        try:
            os.makedirs(save_path)
        except:
            pass
        res.to_csv(os.path.join(save_path, f"UCRL-AC.csv"), sep=" ")
    return res


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_path = os.path.join(dir_path, "experiments", "UCRL-AC", "S20_mu0.3")
    save_path = os.path.join(dir_path, "data", "UCRL-AC", "S20_mu0.3")
    get_dataframe(exp_path, save_path, save=True)
