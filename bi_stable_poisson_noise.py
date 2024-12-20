import numpy as np
import matplotlib.pyplot as plt
#taken from https://sci-hub.se/https://link.springer.com/article/10.1023/A:1012885314187
#and also https://www.sciencedirect.com/science/article/pii/S0896627302010929
#ignored some stuff, they have realy complicated synaptic stuff, I'm just going to use the raw weights with no decay after a spike let me tell you that much friendo

sim_length = 1 #number of seconds in real time
del_t = 1E-3 #in seconds
sim_steps = int(sim_length/del_t) #number of time steps taken

#all volts in mV
V_r = -70 #resting
V_E = 0 #excitatory
V_I = -70 #inhibitory
V_thresh = -50 #threshold
V_reset = -55 #reset
t_refrac = 2E-3
t_slf = 1000 #time since last fire, control refractory period and will allow for proper action potential shaped synapse firing thing you know

#these are, capacitence in nF, leak conductance in nano siemens, refractory period in seconds, and the time constant for synaptic conductance which is currently unuse
excite_neur_params = [0.5, 25, 2E-3, 20]
inhib_neur_params = [0.2, 20, 1E-3, 10]

#these are the volatile neuron parameters, they are the voltage in mV and the time since the neuron last fired, initilized to 100 seconds as the neurons should start not firing
neur_vol = [-70, 100, 0]

#check when inhibition is different size than the regions
#change weights
num_in_region = 1100
num_choices = 2
num_excite = num_in_region*num_choices
num_inhib = 300
num_neurons = num_inhib + num_excite

exin_array = np.zeros(num_neurons)
neurons = np.zeros((num_neurons, 3)) #stores the voltage and the tslf (time since last fire) and if they just fired
neur_params = np.zeros((num_neurons, 4))
syn_weights = np.random.rand(num_neurons, num_neurons)
architecture = np.zeros((num_neurons, num_neurons))
ee_weight = 5.5
ie_weight = 5.5
ei_weight = 2.5

#order is [post_synaptic, pre_synaptic]
jFs = np.zeros((num_neurons, sim_steps))
Vs = np.zeros((num_neurons, sim_steps))

for n in range(num_neurons):
	syn_weights[n,n] = 0 #no self stim boys

	if n < num_excite: 
		exin_array[n] = 1
		neurons[n, :] = neur_vol[:]
		neur_params[n, :] = excite_neur_params[:]
	else:
		exin_array[n] = 0
		neurons[n, :] = neur_vol[:]
		neur_params[n, :] = inhib_neur_params[:]


for i in range(num_choices):
	for pre_syn_idx in range(num_in_region):

		#sets up the connections within a given region
		for post_syn_idx in range(num_in_region):
			true_pre_syn_idx = pre_syn_idx+(i*num_in_region)
			true_post_syn_idx = post_syn_idx+(i*num_in_region)
			if true_post_syn_idx != true_pre_syn_idx: 
				architecture[true_post_syn_idx][true_pre_syn_idx] = ee_weight
		
		#connections of this region exciting the inhibitory region
		for post_syn_idx in range(num_inhib):
			true_pre_syn_idx = pre_syn_idx+(i*num_in_region)
			true_post_syn_idx = post_syn_idx+(num_choices*num_in_region)
			architecture[true_post_syn_idx][true_pre_syn_idx] = ei_weight

	#connections of the inhibitory region inhibiting this region
	for pre_syn_idx in range(num_inhib):
		for post_syn_idx in range(num_in_region):
			
			true_pre_syn_idx = pre_syn_idx+(num_choices*num_in_region)
			true_post_syn_idx = post_syn_idx+(i*num_in_region)
			architecture[true_post_syn_idx][true_pre_syn_idx] = ie_weight

syn_weights = np.multiply(syn_weights, architecture)
np.save("architecture", architecture)
np.save("syn_weights", syn_weights)

def update_net(local_neurons, local_noise):

	local_neurons[:, 1] += np.ones(num_neurons)*del_t

	leak = np.multiply(-1*neur_params[:, 0], local_neurons[:, 0] - V_r)

	excite = np.zeros(num_neurons)
	inhib = np.zeros(num_neurons)

	excite = -1*np.sum(exin_array * local_neurons[:, 2] * syn_weights, axis=1) * (local_neurons[:, 0] - V_E)
	inhib  = -1*np.sum((1 - exin_array) * local_neurons[:, 2] * syn_weights, axis=1) * (local_neurons[:, 0] - V_I)

	#print(np.mean(excite), np.mean(local_noise), np.mean(leak))

	del_v = np.add((leak[:] + excite[:] + inhib[:])*del_t/neur_params[:, 1], local_noise[:])
	local_neurons[:, 0] += del_v
	
	local_neurons[:, 2] = np.zeros(num_neurons)
	fired = np.where(local_neurons[:, 0] > V_thresh)
	for n in fired:
		local_neurons[n, 0] = V_reset
		local_neurons[n, 1] = 0
		local_neurons[n, 2] = 1

	return local_neurons

for t in range(sim_steps):
	#noise = np.random.rand(num_neurons)*0.15 - np.random.normal(0.05, 1, num_neurons)
	#noise = (np.random.rand(num_neurons)>0.9) - np.random.normal(0.05, 1, num_neurons)
	#noise = (np.random.rand(num_neurons)>0.95)*0.5
	#noise = (np.random.rand(num_neurons)>0.8)*0.1
	noise = np.multiply(np.random.rand(num_neurons)>0.8, np.random.rand(num_neurons))*0.2

	neurons = update_net(neurons, noise)
	jFs[:, t] = neurons[:, 2]
	Vs[:, t] = neurons[:, 0]


firing_rates = []
for j in range(num_choices):
	firing_rates.append(np.mean(jFs[num_in_region*j:num_in_region*(j+1), 3*sim_steps//4:]))
print(firing_rates)

plt.figure()
plt.imshow(Vs)

plt.figure()
plt.imshow(jFs)
plt.show()
