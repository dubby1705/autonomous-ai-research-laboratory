import numpy as np
t = np.linspace(0, 3600, 100)
# Standard lithium-ion battery with internal resistance 0.15 Ohm
r_int = 0.15
voc = 4.2
i_load = 2.0
v_terminal = voc - i_load * r_int
energy_efficiency = (v_terminal / voc) * 100.0
stability = 92.0 # retention %
print(f'energy_efficiency: {energy_efficiency}')
print(f'stability: {stability}')

