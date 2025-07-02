# class Plotter:
#     def __init__(self):
#         self.saved_states = {}


import math

import matplotlib.pyplot as plt

from src.utils.standard_funcs import poisson

from .recomb_markov_chain import RecombMarkovChain

fig = plt.figure(figsize=[9, 7])
x_range = range(7)


p_zero_to_zero = 0.5
poisson_line = plt.bar(
    x_range,
    [poisson(i, -math.log(p_zero_to_zero)) for i in x_range],
    width=0.3,
    label='Pois',
)

rmc = RecombMarkovChain(7)
rmc.reset_to_poisson(p_zero_to_zero)
print(rmc)
mc_default_line = plt.bar(
    [x + 0.3 for x in x_range],
    [rmc.get_p_cos(k) for k in x_range],
    width=0.3,
    label='MC Default',
)
_ = plt.bar_label(mc_default_line, padding=3)

rmc2 = RecombMarkovChain(7)
rmc2.reset_to_poisson(p_zero_to_zero)
# rmc2.perturb_transition_probability(1, 2, -0.2)
rmc2.on_all_onward_transitions(rmc2.scale_transition_probability, 0.8, start=1)
print(rmc2)
mc_interf_line = plt.bar(
    [x + 0.6 for x in x_range],
    [rmc2.get_p_cos(k) for k in x_range],
    width=0.3,
    label='MC Interf',
)
_ = plt.bar_label(mc_interf_line, padding=3)

ax = plt.gca()
_ = plt.legend()
plt.savefig('out--overall_interf_markov_chain.png')
plt.show()
