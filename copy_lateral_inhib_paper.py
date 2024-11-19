import numpy as np
import matplotlib.pyplot as plt
#copying this paper as close as I can: https://sci-hub.se/https://doi.org/10.1016/j.neunet.2019.09.007
#the logic of my code works I think but either my LIF model is different or I missed something in the paper or who the hell knows what
#oh wait its probably synapse decay stuff, i bet that whats it they're modeling a real action potential

train_data, train_labels = np.load('mnist_train_ones_zeros_data.npy'), np.load('mnist_train_ones_zeros_labels.npy')

def convert_to_binary_1D_normalized(array_2D, max_probability=1.0):
	flat_array = array_2D.flatten()
	probabilities = flat_array * max_probability
	binary_1D_array = np.random.rand(len(flat_array)) < probabilities
	return binary_1D_array.astype(int)  # Convert boolean to int (0 or 1)

num_sims = 100000
batch_size = 500
sim_length = 350 #number of miliseconds in real time
del_t = 0.5 #in seconds
sim_steps = int(sim_length/del_t) #number of time steps taken

#all volts in mV
Tau = 100
V_r = -65 #resting
V_E = 0 #excitatory
V_I = -100 #inhibitory
V_thresh = -72 #threshold
V_reset = -65 #reset
t_refrac = 2
input_prob = 0.025

num_input = 784
num_hidden_exc = 400
num_hidden_inhib = num_hidden_exc
num_out = 10
num_neurons = num_hidden_exc + num_hidden_inhib + num_out
num_all = num_neurons + num_input

#these are, capacitence in nF, leak conductance in nano siemens, and the time constant for synaptic conductance which is currently unuse, then the threshold increase which is dynamic in this paper
excite_neur_params = [0.5, 25, 20, 20]
inhib_neur_params = [0.2, 20, 10, 20]

#these are the volatile neuron parameters, they are the voltage in mV and the time since the neuron last fired, initilized to 100 seconds as the neurons should start not firing
neur_vol = [-70, 100, 0]

exin_array = np.ones(num_all)
synapses = np.zeros((num_neurons, num_all))
membrane_volts = np.ones(num_neurons)*V_r
g_E = np.zeros(num_neurons)
g_I = np.zeros(num_neurons)
fired = np.zeros(num_all)
tslfs = np.ones(num_all)*1000
neur_params = np.zeros((num_neurons, 4))
jFs = np.zeros((num_all, sim_steps))
volts = np.zeros((num_neurons, sim_steps))

idx_pairs = []
for pre_syn_idx in range(num_all):
	adjusted_pre_syn_idx = pre_syn_idx - num_input
	if adjusted_pre_syn_idx < 0:
		adjusted_pre_syn_idx = num_all + 1000
	for post_syn_idx in range(num_neurons):
		if pre_syn_idx < num_input and post_syn_idx < num_hidden_exc: #input to excitatory hidden
			synapses[post_syn_idx, pre_syn_idx] = np.random.rand()*0.3
			idx_pairs.append([pre_syn_idx, post_syn_idx])

		if adjusted_pre_syn_idx == post_syn_idx - num_hidden_exc and post_syn_idx > num_hidden_exc and post_syn_idx < num_hidden_exc + num_hidden_inhib: #hidden excite to corresponding hidden inhib
			synapses[post_syn_idx, pre_syn_idx] = 30 #the corresponding neuron should always fire, right?

		if post_syn_idx < num_hidden_exc and adjusted_pre_syn_idx > num_hidden_exc and adjusted_pre_syn_idx < num_hidden_exc + num_hidden_inhib and adjusted_pre_syn_idx != post_syn_idx: #hidden inhib to all hidden excite
			synapses[post_syn_idx, pre_syn_idx] = np.random.rand()*0.3
			idx_pairs.append([pre_syn_idx, post_syn_idx])

		if adjusted_pre_syn_idx < num_hidden_exc and post_syn_idx > num_hidden_exc + num_hidden_inhib: #hidden excite to the output neurons
			synapses[post_syn_idx, pre_syn_idx] = np.random.rand()*0.3
			idx_pairs.append([pre_syn_idx, post_syn_idx])

for n in range(num_neurons):
	if n < num_hidden_exc or n > num_hidden_exc + num_hidden_inhib:
		neur_params[n, :] = excite_neur_params[:]
	else:
		exin_array[n + num_input] = 0
		neur_params[n, :] = inhib_neur_params[:]

def update_net(local_tslfs, local_mem_volt, local_syn_weights, local_fired, local_g_E, local_g_I):

	local_tslfs += np.ones(num_all)*del_t

	leak = V_r - local_mem_volt[:]

	excite = np.sum(exin_array * local_fired[:] * local_syn_weights, axis=1)
	inhib  = np.sum((1 - exin_array) * local_fired[:] * local_syn_weights, axis=1)
	
	local_g_E[:] += del_t * (-local_g_E[:] + excite)
	local_g_I[:] += del_t * (-local_g_I[:] + inhib)

	del_v = (leak[:] + local_g_E[:] * (V_E - local_mem_volt[:]) + local_g_I[:] * (V_I - local_mem_volt[:])) * del_t / Tau

	local_mem_volt[:] += del_v * (local_tslfs[num_input:] > np.ones(num_neurons)*t_refrac)

	local_tslfs[:num_input] -= local_tslfs[:num_input]*local_fired[:num_input]

	local_fired[:] = np.zeros(num_all)

	just_fired = np.where(local_mem_volt[:] > V_thresh + neur_params[:,3])

	for n in just_fired:
		local_mem_volt[n] = V_reset
		local_tslfs[num_input + n] = 0
		local_fired[num_input + n] = 1
	return local_tslfs, local_mem_volt, local_fired, local_g_E, local_g_I

train_data, train_labels = np.load('mnist_train_ones_zeros_data.npy'), np.load('mnist_train_ones_zeros_labels.npy')

data_idx = np.random.randint(10000)
for s in range(sim_steps):
	fired[:num_input] = convert_to_binary_1D_normalized(train_data[data_idx][0], input_prob)
	jFs[:, s] = fired[:]
	volts[:, s] = membrane_volts[:]
	tslfs, membrane_volts, fired, g_E, g_I = update_net(tslfs, membrane_volts, synapses, fired, g_E, g_I)

print(np.sum(jFs[:num_input,:])/sim_steps)
print(np.sum(jFs[num_input:,:]))
print(np.sum(jFs[num_input+num_hidden_exc+num_hidden_inhib:,:]))
plt.figure()
plt.imshow(jFs)
plt.figure()
plt.imshow(volts)
plt.show()

