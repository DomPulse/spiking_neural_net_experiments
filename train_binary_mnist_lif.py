import numpy as np
import matplotlib.pyplot as plt
#taken from https://sci-hub.se/https://link.springer.com/article/10.1023/A:1012885314187
#and also https://www.sciencedirect.com/science/article/pii/S0896627302010929
#ignored some stuff, they have realy complicated synaptic stuff, I'm just going to use the raw weights with no decay after a spike let me tell you that much friendo

def convert_to_binary_1D_normalized(array_2D, max_probability=1.0):
	flat_array = array_2D.flatten()
	inverted_probabilities = 1 - flat_array
	probabilities = inverted_probabilities * max_probability
	binary_1D_array = np.random.rand(len(flat_array)) < probabilities
	return binary_1D_array.astype(int)  # Convert boolean to int (0 or 1)

num_sims = 10000
sim_length = 0.6 #number of seconds in real time
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
num_inputs = 784
num_nuer_a_in = num_inputs + num_neurons #number of nuerons and inputs

exin_array = np.zeros(num_neurons)
neurons = np.zeros((num_neurons, 3)) #stores the voltage and the tslf (time since last fire) and if they just fired
re_init_neur = np.zeros((num_neurons, 3)) 
neur_params = np.zeros((num_neurons, 4))

syn_weights = np.random.rand(num_neurons, num_neurons)
in_to_excite = np.random.rand(num_inputs, num_excite)

architecture = np.zeros((num_neurons, num_neurons))
ee_weight = 5.3
ie_weight = 5.5
ei_weight = 2.4
des_mean_sig = 0.01
look_back = 20

idx_pairs = []

#order is [post_synaptic, pre_synaptic]
jFs = np.zeros((num_nuer_a_in, sim_steps))

for n in range(num_neurons):
	syn_weights[n,n] = 0 #no self stim boys
	re_init_neur[n, :] = neur_vol[:]

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
			idx_pairs.append([true_pre_syn_idx, true_post_syn_idx])

syn_weights = np.multiply(syn_weights, architecture)

np.save("architecture", architecture)
np.save("init_syn_weights", syn_weights)
np.save("init_in_to_excite", in_to_excite)

def find_nearest(array, value):
	array = np.asarray(array)
	try:
		idx = (np.abs(array - value)).argmin()
		return array[idx]
	except:
		print(array, value)
		return(np.nan())

def stdp_but_faster(r, spike_idxs):
	#boy i sure do hope the indexing is right, if only i was smart enough to know how my own code works
	delta_stdp_syn = np.zeros((num_neurons, num_neurons))
	delta_stdp_in = np.zeros((num_inputs, num_excite))

	for pair in idx_pairs:
		pre_syn_idx = pair[0]
		post_syn_idx = pair[1]
		for this_spike_time in spike_idxs[num_inputs + post_syn_idx]:
			
			if np.size(spike_idxs[num_inputs + pre_syn_idx]) != 0:
				delta_t = this_spike_time - find_nearest(spike_idxs[num_inputs + pre_syn_idx], this_spike_time)
				if delta_t < 0:
					direction = -1
					delta_t *= -1
				else:
					direction = 1
				delta_stdp_syn[post_syn_idx][pre_syn_idx] += direction*r*np.exp(-1*delta_t/look_back)*(2*exin_array[pre_syn_idx]-1)*2/look_back

	for pre_syn_idx in range(num_inputs):
		for post_syn_idx in range(num_excite):
			for this_spike_time in spike_idxs[num_inputs + post_syn_idx]:
				if np.size(spike_idxs[pre_syn_idx]) != 0:
					delta_t = this_spike_time - find_nearest(spike_idxs[pre_syn_idx], this_spike_time)
					if delta_t < 0:
						direction = -1
						delta_t *= -1
					else:
						direction = 1
					#PAY ATTENTION TO THIS DUMMY
					delta_stdp_in[pre_syn_idx][post_syn_idx] += direction*r*np.exp(-1*delta_t/look_back)*2/look_back
					#boy i sure do hope the indexing is right, if only i was smart enough to know how my own code works


	delta_stdp_syn = np.clip(delta_stdp_syn, -0.25, 0.25)
	delta_stdp_in = np.clip(delta_stdp_in, -0.05, 0.05)
	return delta_stdp_syn, delta_stdp_in

def update_net(local_neurons, local_noise):

	#local_neurons[:, 1] += np.ones(num_neurons)*del_t

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
		#local_neurons[n, 1] = 0
		local_neurons[n, 2] = 1

	return local_neurons

train_data, train_labels = np.load('mnist_train_ones_zeros_data.npy'), np.load('mnist_train_ones_zeros_labels.npy')

total_right = 0
for sim in range(num_sims):
	data_idx = np.random.randint(0, len(train_data))
	init_mean_sig = np.mean(np.matmul(convert_to_binary_1D_normalized(train_data[data_idx][0], max_probability=0.25), in_to_excite))
	times_nuer_fire = []
	for t in range(sim_steps):
		#noise = np.multiply(np.random.rand(num_neurons)>0.8, np.random.rand(num_neurons))*0.2
		flat_in = convert_to_binary_1D_normalized(train_data[data_idx][0], max_probability=0.05)
		sig = np.matmul(flat_in, in_to_excite)
		sig /= (init_mean_sig/des_mean_sig)
		noise = np.random.normal(0.05, 1, num_neurons)

		all_in = noise
		all_in[0:num_excite] += sig[:]

		neurons = update_net(neurons, all_in)
		jFs[:784, t] = (flat_in > 0)[:]
		jFs[784:, t] = neurons[:, 2]

	for n in range(num_nuer_a_in):
		times_nuer_fire.append(np.where(jFs[n, :] == 1)[0])

	firing_rates = []
	for j in range(num_choices):
		firing_rates.append(np.mean(jFs[num_inputs + num_in_region*j:num_inputs + num_in_region*(j+1), 3*sim_steps//4:]))

	
	right = firing_rates[train_labels[data_idx]] == np.max(firing_rates)
	total_right += right
	reward = 2*(firing_rates[train_labels[data_idx]] == np.max(firing_rates)) - 1
	print(firing_rates, right, total_right/(sim+1))

	delta_syn_weights, delta_in_weights = stdp_but_faster(reward, times_nuer_fire)
	#print(np.mean(delta_syn_weights), np.mean(delta_in_weights))
	
	pre_ie_mean = np.mean(syn_weights[0:num_excite,num_excite:num_neurons])
	syn_weights += delta_syn_weights
	syn_weights = np.clip(syn_weights, 0, 10)
	new_ie_mean = np.mean(syn_weights[0:num_excite,num_excite:num_neurons])
	syn_weights[0:num_excite,num_excite:num_neurons] *= pre_ie_mean/new_ie_mean

	pre_in_mean = np.mean(in_to_excite)
	in_to_excite += delta_in_weights
	in_to_excite = np.clip(in_to_excite, 0, 10)
	new_in_mean = np.mean(in_to_excite)
	in_to_excite *= pre_in_mean/new_in_mean

	neurons[::] = re_init_neur[::]

	if sim%10 == 0:
		version = "syn_weights_" + str(sim)
		np.save(version, syn_weights)
		version = "in_to_excite" + str(sim)
		np.save(version, in_to_excite)
