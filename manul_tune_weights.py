
import numpy as np
import matplotlib.pyplot as plt
import time

init_neuron = [-65.0, -13.0, 0.0, 0.0] #V, u, I, jF for the neuron at the start of sim
num_neurons = 1000
frac_connected = np.min([1, 1000/num_neurons])
sim_length = 3000
delay_show = 50


neurons = np.zeros((num_neurons, 4))
alphabet = np.zeros((num_neurons, 4))
glob_syn_array = np.zeros((num_neurons, num_neurons))
architecture = np.zeros((num_neurons, num_neurons))
exin_array = np.zeros(num_neurons)
jFs = np.zeros((num_neurons, sim_length))

ee_weight = 0.6
ie_weight = 0.2
ei_weight = 0.2
frac_inhib = 0.05
poisson_input_freq = 20

#creates the inital conditions and the fixed fit parameters for the neurons
for pre_syn_idx in range(0, num_neurons):
	if pre_syn_idx > num_neurons*frac_inhib:
		exin_array[pre_syn_idx] = 1
		re = np.random.rand()
		alphabet[pre_syn_idx] = [0.02, 0.2, -65+15*re**2, 8-6*re**2]
		neurons[pre_syn_idx] = [-65.0, -65.0*alphabet[pre_syn_idx][1], 0.0, 0.0]
	else:
		exin_array[pre_syn_idx] = -1 
		ri = np.random.rand()
		alphabet[pre_syn_idx] = [0.02+0.08*ri, 0.25-0.05*ri, -65, 2]
		neurons[pre_syn_idx] = [-65.0, -65.0*alphabet[pre_syn_idx][1], 0.0, 0.0]

#creates the synaptic array connecting neurons
for post_syn_idx in range(0, num_neurons):
	for pre_syn_idx in range(0, num_neurons):

		if exin_array[post_syn_idx] == 1 and exin_array[pre_syn_idx] == 1:
			glob_syn_array[post_syn_idx][pre_syn_idx] = ee_weight

		if exin_array[post_syn_idx] == -1 and exin_array[pre_syn_idx] == 1:
			glob_syn_array[post_syn_idx][pre_syn_idx] = ei_weight

		if exin_array[post_syn_idx] == 1 and exin_array[pre_syn_idx] == -1:
			glob_syn_array[post_syn_idx][pre_syn_idx] = ie_weight

		if pre_syn_idx == post_syn_idx or (exin_array[post_syn_idx] == -1 and exin_array[pre_syn_idx] == -1):
			glob_syn_array[post_syn_idx][pre_syn_idx] = 0

#creates the model architecture instead of one big dense mess
idx_pairs = []
for pre_syn_idx in range(num_neurons):
	for post_syn_idx in range(num_neurons):
		#sets up the connections within a given region
		if pre_syn_idx != post_syn_idx and np.random.rand() < frac_connected: 
			architecture[post_syn_idx][pre_syn_idx] = 1
			idx_pairs.append([pre_syn_idx, post_syn_idx])

glob_syn_array = np.multiply(architecture, glob_syn_array)
mask = (np.multiply(exin_array, np.ones(num_neurons))+1)/2

np.save("initial_syn_weights", glob_syn_array)
np.save("alphabet", alphabet)
np.save("exin_array", exin_array)
np.save("architecture", architecture)

def update_network(t):
	#currents_ex = np.multiply(5*np.random.normal(0, 1, num_neurons), mask)
	#currents_in = np.multiply(2*np.random.normal(0, 1, num_neurons), np.ones(num_neurons)-mask)
	#neurons[:, 2] += currents_in + currents_ex
	
	neurons[:, 2] += (np.random.rand(num_neurons) < poisson_input_freq/1000)*10
	neurons[:, 2] += np.matmul(glob_syn_array, np.multiply(neurons[:, 3], exin_array))

	neurons[:, 0] += 0.5*((0.04*(neurons[:, 0]**2))+(5*neurons[:, 0])+140-neurons[:, 1]+neurons[:, 2])
	neurons[:, 0] += 0.5*((0.04*(neurons[:, 0]**2))+(5*neurons[:, 0])+140-neurons[:, 1]+neurons[:, 2])
	neurons[:, 1] += alphabet[:, 0]*(alphabet[:, 1]*neurons[:, 0] - neurons[:, 1])
	neurons[:, 3] = 0
	
	#try to improve this to see if it goes in "real time"
	neurons_that_fired = np.where(neurons[:, 0] > 30)

	for n in neurons_that_fired[0]:
		neurons[n][0] = alphabet[n][2]
		neurons[n][1] += alphabet[n][3]
		neurons[n][3] = 1 

	neurons[:, 2] = np.zeros(num_neurons)

	
	#return neurons_that_fired
	
Vs = np.zeros(sim_length)
activity = np.zeros(sim_length)
#print(exin_array)
for t in range(sim_length):
	#if t > sim_length/3 and t < 2*sim_length/3:
	#	neurons[:, 2] = neurons[:, 2] = 7
	update_network(t)
	Vs[t] = neurons[0, 0]
	jFs[:, t] = neurons[:, 3]
	if t > 50:
		activity[t] = np.mean(jFs[:, t-50:t])

plt.imshow(jFs)
plt.show()

plt.plot(np.linspace(delay_show, sim_length, sim_length-delay_show), activity[delay_show:])
plt.show()