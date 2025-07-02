import matplotlib.pyplot as plt

from src.utils.standard_funcs import poisson

fig = plt.figure(figsize=[9, 7])
x_range = range(10)


def restacked_poisson(x: float, lam: float) -> float:
    if x < 1:
        return poisson(x, lam)
    if x == 1:
        return poisson(x, lam) + 0.2 * poisson(x + 1, lam)
    else:
        return 0.8 * poisson(x, lam) + 0.2 * poisson(x + 1, lam)


def modded_1to2_poisson(x: float, lam: float) -> float:
    if x < 1:
        return poisson(x, lam)
    if x == 1:
        return poisson(x, lam) + 0.2 * (1 - poisson(0, lam) - poisson(1, lam))
    else:
        return 0.8 * poisson(x, lam)


avg = 0.7
poisson_line = plt.bar(x_range, [poisson(i, avg) for i in x_range], label='Pois')
modded_poisson_line = plt.bar(
    x_range,
    [restacked_poisson(i, avg) for i in x_range],
    width=0.6,
    label='Restacked Pois',
)
markov_line = plt.bar(
    x_range,
    [modded_1to2_poisson(i, avg) for i in x_range],
    width=0.4,
    label='Modded 1->2 Pois',
)

ax = plt.gca()
_ = plt.legend()
plt.savefig('out--plot_poisson_mod_formula.png')
plt.show()
