import numpy as np
import matplotlib.pyplot as plt

Tau = 100
E_rest = -65
E_reset = -65
E_exc = 0
E_Inhib = -100
V_thresh = -52

V = -60

dt = 0.5
sim_length = 350

g_E = 0
g_I = 0

t = 0

Vs = []
while t < sim_length:
	t += dt
	V += dt * ((E_rest - V) + g_E * (E_exc - V) + g_I * (E_Inhib - V)) / Tau
	g_E += dt * (-g_E + 0.15 * np.random.rand()*2)
	g_I += dt * (-g_I + 0.15 * np.random.rand()*0.5)
	print(g_E, 0.15 * np.random.rand()*2)
	if V >= V_thresh:
		V = E_reset
	Vs.append(V)

plt.plot(Vs)
plt.show()
