import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(9.0, 7.0))
x_range = np.linspace(0, 3, 100)


# haldane mapping func, with p exposed as parameter
def H(x, p):
    return (1 - np.exp(-1 * p * x)) / p


y = H(x_range, 2)
plt.plot(x_range, y)
for num in np.linspace(0.1, 2, 19):
    y = H(x_range, num)
    plt.plot(x_range, y)
plt.plot(np.linspace(0, 2.5, 100), np.linspace(0, 2.5, 100))
plt.savefig('out--plot_haldane_fn_with_diff_p.png')
plt.show()
