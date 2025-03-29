import numpy as np
import matplotlib.pyplot as plt

#taken from this bad boi https://journals.physiology.org/doi/epdf/10.1152/jn.00641.2003
#they explicity set the membrane surface area which I should change to reflect morphology when I integrate with the Allen data
#the only thing they vary is the conductance, I geuss I could vary more but let's copy them first
#all voltages in mV, all times in ms
#currents should be in nA

def m_or_h_infin(V, top, bottom, Ca_adj = False, Ca = 1):
	mult = 1
	if Ca_adj:
		mult = Ca/(Ca + 3)
	denom = 1 + np.exp((V + top)/bottom)
	return mult/denom

def most_taus(V, numer, top, bottom, offset):
	denom = 1 + np.exp((V + top)/bottom)
	return offset + numer/denom

def double_exp_taus(V, numer, top_1, bottom_1, top_2, bottom_2, offset):
	denom = np.exp((V + top_1)/bottom_1) + np.exp((V + top_2)/bottom_2)
	return offset + numer/denom

def tau_h_Na(V):
	first_mult = 1.34/(1+np.exp((V + 62.9)/(-10)))
	second_mult = 1.5 + 1/(1 + np.exp((V + 34.9)/3.6))
	return first_mult*second_mult

def step_m_or_h(j, tau, infin):
	#j can be m or h, tau should be sef explanatory, infin is m or h infinity
	dj = time_step*(infin - j)/tau
	return dj

def step_Ca2(I_CaT, I_CaS, Ca2):
	#just realized they don't number their equations in this paper, weird
	dCa2 = time_step*(-14.96*(I_CaT + I_CaS) - Ca2 + 0.05)/200
	return dCa2

def current_contrib(V, E, g, m, p, h = 1):
	#hey some of the ions dont have h modeles in the table 1, gonna just assume constant 1 for those until i see otherwise
	I = g*(m**p)*h*(V-E)
	return I


A = 6.28E-4 #area in cm^2
C = 6.28E-10 #capacitence in nF, something tells me they picked this to mostly cancel with area
I_ext = 0
time_step = 0.5
sim_length = 1000
sim_steps = int(sim_length/time_step)

Na_g = 100
CaT_g = 0
CaS = 10
A_g = 50
KCa_g = 20
Kd_g = 100
H_g = 0.04
leak = 0.02

#volatile initialization
V_membrane = -65
Ca2 = 0.05

#I genuinely don't know if the time constants are meant to be constant, like yeah they should be but they are functions of V so like?
#I think they do change bc of top of page 3 but like 
#conductance, m, h, p, tau_m, tau_h, then reversal potential
Na_params = [Na_g, m_or_h_infin(V_membrane, 25.5, -5.29), m_or_h_infin(V_membrane, 48.9, 5.18), 3, most_taus(V_membrane, -2.52, 120, -25, 2.64), tau_h_Na(V_membrane), 50]
CaT_params = [CaT_g, m_or_h_infin(V_membrane, 27.1, -7.2), m_or_h_infin(V_membrane, 32.1, 5.5), 3, most_taus(V_membrane, -42.6, 68.1, -20.5, 43.4), most_taus(V_membrane, -179.6, 55, -16.9, 210), ]
CaS_params = [CaS, m_or_h_infin(V_membrane, 33, -8.1), m_or_h_infin(V_membrane, 60, 6.2), 3, double_exp_taus(V_membrane, 14, 27, 10, 70, -13, 2.8), double_exp_taus(V_membrane, 300, 55, 9, 65, -16, 120)]
A_params = [A_g, m_or_h_infin(V_membrane, 27.2, -8.7), m_or_h_infin(V_membrane, 56.9, 4.9), 3, most_taus(V_membrane, -20.8, 32.9, -15.2, 23.2), most_taus(V_membrane, -58.4, 38.9, -26.5, 77.2)]
KCa_params = [KCa_g, m_or_h_infin(V_membrane, 28.3, -12.6, True, Ca2), 1, 4, most_taus(V_membrane, -150.2, 46, -22.7, 180.6), 1]
Kd_params = [Kd_g, m_or_h_infin(V_membrane, 12.3, -11.8), 1, 4, most_taus(V_membrane, -12.8, 28.3, -19.2, 14.4), 1]
H_params = [H_g, m_or_h_infin(V_membrane, 75, 5.5), 1, 1, double_exp_taus(V_membrane, 2, 169.7, -11.6, -26.7, 14.3, 0), 1]

all_params = np.asarray([Na_params, CaT_params, CaS_params, A_params, KCa_params, Kd_params, H_params])
print(all_params)

for s in range(sim_steps):
	current_sum = 0
	for params in all_params:
		current_sum += current_contrib(V)
	dV = time_step*(current_sum + I_ext)/C