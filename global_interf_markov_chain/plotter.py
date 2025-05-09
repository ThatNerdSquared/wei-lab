# class Plotter:
#     def __init__(self):
#         self.saved_states = {}


import matplotlib.pyplot as plt
import numpy as np
import math
from recomb_markov_chain import RecombMarkovChain

fig = plt.figure(figsize=[9, 7])
x_range = range(7)

def poisson(x, lam):
    return ((lam ** x) * np.exp(-1 * lam)) / math.gamma(x + 1)

p_zero_to_zero = 0.5
poisson_line = plt.bar(x_range, [poisson(i, -math.log(p_zero_to_zero)) for i in x_range], width=0.3, label="Pois")

rmc = RecombMarkovChain(7)
rmc.reset_to_poisson(p_zero_to_zero)
print(rmc)
mc_default_line = plt.bar([x + 0.3 for x in x_range], [rmc.get_p_cos(k) for k in x_range], width=0.3, label="MC Default")
plt.bar_label(mc_default_line, padding=3)

rmc2 = RecombMarkovChain(7)
rmc2.reset_to_poisson(p_zero_to_zero)
rmc2.perturb_transition_probability(1, 2, -0.2)
print(rmc2)
mc_interf_line = plt.bar([x + 0.6 for x in x_range], [rmc2.get_p_cos(k) for k in x_range], width=0.3, label="MC Interf")
plt.bar_label(mc_interf_line, padding=3)

ax = plt.gca()
plt.legend()
plt.savefig("out--global_interf_markov_chain.png")
plt.show()
