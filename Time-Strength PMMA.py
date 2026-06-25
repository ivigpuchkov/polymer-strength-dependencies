import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats

def time_strength(t, tau, U_0, gamma):
    k = 1.38e-23
    T = 293
    tau = tau * 1e-6
    t_0 = 1e-13
    U_0 = U_0 * 1000 / 6e23
    gamma = gamma * 1e-27
    val_true = 2 * tau * (U_0 / gamma - (k * T / gamma) * np.log(t / t_0)) / t
    val_false = (U_0 / gamma - (k * T / gamma) * np.log(t / t_0)) / (1 - (tau / (2 * t)))
    return np.where(t < tau, val_true, val_false)

exp_data = {
    90.0:  [0.060, 5e-4],
    104.0: [0.060, 5e-3],
    129.0: [0.085, 5e-2],
    155.0: [0.100, 1.0],
    195.0: [0.100, 10.0],
    219.0: [0.100, 50.0],
    284.0: [0.060, 2500.0]
} # {Макс. напряжение : [деформация, скорость деформации]}

sigmas = np.array(list(exp_data.keys())) * 1e6
epsilons = np.array([val[0] for val in exp_data.values()])
dot_epsilons = np.array([val[1] for val in exp_data.values()])
n = len(exp_data)
t_exp = epsilons / dot_epsilons

p0_array = [1, 1, 1]
upper_bounds = [1000, 500, 1000]
lower_bounds = [0.001, 1, 0.001]

epsilon = np.inf
popt, pcov = curve_fit(time_strength, t_exp, sigmas, p0=p0_array,
        bounds=(lower_bounds, upper_bounds))

confidence = 0.95
alpha = 0.05 # квантиль 0.975
df = n - len(p0_array)
t_multiplier = stats.t.ppf(1.0 - alpha / 2, df)
diag_cov = np.diag(pcov)

print(f"Параметр tau = {popt[0]} +- {t_multiplier * np.sqrt(diag_cov[0])} мкс")
print(f"Параметр U_0 = {popt[1]} +- {t_multiplier * np.sqrt(diag_cov[1])} кДж/моль")
print(f"Параметр gamma = {popt[2]} +- {t_multiplier * np.sqrt(diag_cov[2])} 10^-27 м^3")