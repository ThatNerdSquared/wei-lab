import matplotlib.pyplot as plt
import numpy as np

from src.utils.standard_funcs import haldane_input_cM

fig = plt.figure(figsize=(9.0, 7.0))
x_range = np.linspace(0, 3, 100)


y = haldane_input_cM(x_range, 2)
_ = plt.plot(x_range, y)

num: np.float64
for num in np.linspace(0.1, 2, 19):
    y = haldane_input_cM(x_range, num)
    _ = plt.plot(x_range, y)
_ = plt.plot(np.linspace(0, 2.5, 100), np.linspace(0, 2.5, 100))
plt.savefig('out--plot_haldane_fn_with_diff_p.png')
plt.show()
