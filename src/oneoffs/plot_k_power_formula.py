import math

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=[9, 7])
x_range = np.linspace(0, 10, 100)


def poisson(x, lam):
    return ((lam**x) * np.exp(-1 * lam)) / math.gamma(x + 1)


poisson_line = plt.plot(x_range, [poisson(i, 0.3) for i in x_range], label='Poisson')

k = poisson(1, 0.3)
custom_line = plt.plot(x_range, [k**i for i in x_range], label='Custom')

diff_line = plt.plot(
    x_range, [(k**i) / poisson(i, 0.3) for i in x_range], label='Custom / poisson'
)

ax = plt.gca()
ax.set_ylim([0, 10])
plt.legend()
plt.savefig('out--plot_k_power_formula.png')
plt.show()
