import numpy as np
import matplotlib.pyplot as plt
#copying this paper as close as I can: https://sci-hub.se/https://doi.org/10.1016/j.neunet.2019.09.007

train_data, train_labels = np.load('binary_mnist_train_data.npy'), np.load('binary_mnist_train_labels.npy')

def convert_to_binary_1D_normalized(array_2D, max_probability=1.0):
	flat_array = array_2D.flatten()
	probabilities = flat_array * max_probability
	binary_1D_array = np.random.rand(len(flat_array)) < probabilities
	return binary_1D_array.astype(int)  # Convert boolean to int (0 or 1)

#hyperparameters or something
epoch = 1000
batch_size_train = 25
batch_size_test = 10
learning_rate = 0.01
beta = 0.1

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
input_prob = 0.035

Tau_theta = 6E6
alpha = 8.4E5
theta_init = 20

num_input = 784
num_hidden_exc = 400
num_hidden_inhib = num_hidden_exc
num_out = 30
num_class = 2
num_neurons = num_hidden_exc + num_hidden_inhib + num_out
num_all = num_neurons + num_input

#these are, capacitence in nF, leak conductance in nano siemens, and the time constant for synaptic conductance which is currently unuse, then the threshold increase which is dynamic in this paper
excite_neur_params = [0.5, 25, 20, theta_init]
inhib_neur_params = [0.2, 20, 10, theta_init]

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
stdp_jFs = np.zeros((num_all, sim_steps))
real_jFs = np.zeros((num_all, sim_steps))
volts = np.zeros((num_neurons, sim_steps))
look_back = 20
idx_pairs = []

def find_nearest(array, value):
	array = np.asarray(array)
	try:
		idx = (np.abs(array - value)).argmin()
		return array[idx]
	except:
		print(array, value)
		return(np.nan())

def firing_rate_update(jFs, target):
	des_fire_rate = np.mean(jFs[:num_input, :])
	#print(np.mean(jFs[:num_input, :]), np.mean(jFs[num_input:num_input+num_hidden_exc, :]), np.mean(jFs[num_all-(target+1), :]))
	excite_return = (2*np.mean(jFs[num_input:num_input+num_hidden_exc, :], axis = 1))/des_fire_rate
	out_return = np.ones(num_out)
	for n in range(num_out):
		if n%num_class == target:
			out_return[n] = (3*np.mean(jFs[num_all-(n+1), :]))/des_fire_rate

	returned = np.ones(num_neurons)
	returned[:num_hidden_exc] = excite_return[:]
	returned[num_hidden_exc+num_hidden_inhib:] = out_return[:]
	returned = np.clip(returned, 0.9, 1.1)
	return returned

def stdp_but_faster(spike_idxs):
	#boy i sure do hope the indexing is right, if only i was smart enough to know how my own code works
	del_synapses = np.zeros((num_neurons, num_all))
	for pair in idx_pairs:
		pre_syn_idx = pair[0]
		post_syn_idx = pair[1]
		for this_spike_time in spike_idxs[num_input + post_syn_idx]:
			if np.size(spike_idxs[pre_syn_idx]) != 0:
				delta_t = this_spike_time - find_nearest(spike_idxs[pre_syn_idx], this_spike_time)
				del_synapses[post_syn_idx][pre_syn_idx] += learning_rate*(1 + 0.5*(delta_t > 0))*np.exp(-1*np.abs(delta_t)/(look_back + 20*(delta_t < 0)))
	return del_synapses

for pre_syn_idx in range(num_all):
	adjusted_pre_syn_idx = pre_syn_idx - num_input
	if adjusted_pre_syn_idx < 0:
		adjusted_pre_syn_idx = num_all + 1000
	for post_syn_idx in range(num_neurons):
		if pre_syn_idx < num_input and post_syn_idx < num_hidden_exc: #input to excitatory hidden
			synapses[post_syn_idx, pre_syn_idx] = np.random.rand()*0.3
			idx_pairs.append([pre_syn_idx, post_syn_idx])

		if post_syn_idx < num_hidden_exc and adjusted_pre_syn_idx > num_hidden_exc and adjusted_pre_syn_idx < num_hidden_exc + num_hidden_inhib and adjusted_pre_syn_idx != post_syn_idx: #hidden inhib to all hidden excite
			synapses[post_syn_idx, pre_syn_idx] = np.random.rand()*0.3
			idx_pairs.append([pre_syn_idx, post_syn_idx])

		if adjusted_pre_syn_idx < num_hidden_exc and post_syn_idx >= num_hidden_exc + num_hidden_inhib: #hidden excite to the output neurons
			synapses[post_syn_idx, pre_syn_idx] = np.random.rand()*0.3
			idx_pairs.append([pre_syn_idx, post_syn_idx])
idx_pairs = np.array(idx_pairs)


def find_input_means():
	#this was written by chatgpt, consult for debug
	# Create an array to store the mean inputs
	input_means = np.full(num_neurons, beta)  # Default to beta if no inputs exist

	# Extract indices for postsynaptic and presynaptic neurons
	post_syn_indices = idx_pairs[:, 1]
	pre_syn_indices = idx_pairs[:, 0]

	# Group by postsynaptic index to compute the mean
	for post_syn_idx in np.unique(post_syn_indices):
		mask = post_syn_indices == post_syn_idx
		pre_indices = pre_syn_indices[mask]

		if pre_indices.size > 0:
			mean = np.mean(synapses[post_syn_idx, pre_indices])
			input_means[post_syn_idx] = mean

	return input_means

for n in range(num_neurons):
	if n < num_hidden_exc or n > num_hidden_exc + num_hidden_inhib:
		neur_params[n, :] = excite_neur_params[:]
	else:
		exin_array[n + num_input] = 0
		neur_params[n, :] = inhib_neur_params[:]

def update_net(local_tslfs, local_mem_volt, local_neur_params, local_syn_weights, local_fired, local_g_E, local_g_I):

	local_tslfs += np.ones(num_all)*del_t

	leak = V_r - local_mem_volt[:]

	excite = np.sum(exin_array * local_fired[:] * local_syn_weights, axis=1)
	inhib  = np.sum((1 - exin_array) * local_fired[:] * local_syn_weights, axis=1)
	
	local_g_E[:] += del_t * (-local_g_E[:] + excite)
	local_g_I[:] += del_t * (-local_g_I[:] + inhib)

	del_v = (leak[:] + local_g_E[:] * (V_E - local_mem_volt[:]) + local_g_I[:] * (V_I - local_mem_volt[:])) * del_t / Tau

	local_mem_volt[:] += del_v * (local_tslfs[num_input:] > np.ones(num_neurons)*t_refrac)

	local_tslfs[:num_input] -= local_tslfs[:num_input]*local_fired[:num_input]

	local_fired[num_input:] = np.zeros(num_neurons)

	just_fired = np.where(local_mem_volt[:] > V_thresh + local_neur_params[:,3])
	for n in just_fired[0]:
		local_mem_volt[n] = V_reset
		local_tslfs[num_input + n] = 0
		local_fired[num_input + n] = 1

		if n < num_hidden_exc:
			local_fired[num_input + n + num_hidden_exc] = 1 #fires the corresponding inhibitroy neurons

	
	if training:
		#this function looks weird in the paper and might not be exactly how they impliment it, check here for bugs

		#this is what is in the paper as best i can tell 
		#local_neur_params[:,3] += del_t * (-1*local_neur_params[:,3] + local_fired[num_input:]*alpha*theta_init/np.abs(2*local_neur_params[:,3] - theta_init))/Tau_theta

		#but i'm writing my own that keeps the core idea of an exponential decay and increases weights as fired
		local_neur_params[:,3] += del_t * (-1*local_neur_params[:,3] + fired[num_input:]*1000)/(5000*Tau)
		np.clip(local_neur_params[:,3], 10, 30)
		#the constants here are just kind of emperically hand tuned, nothing too serious
	
	return local_tslfs, local_mem_volt, local_neur_params, local_fired, local_g_E, local_g_I

#pretrained kinda, just need to tune the firing rates right
#synapses = np.load("lateral_inhib_mostly_same\\synapses_130.npy")

training = True
batch_size = batch_size_train
print("gay ming")
for e in range(1, epoch):
	
	num_right = 0		

	buffer_delta_syn_weights = np.zeros((num_neurons, num_all))
	buffer_delta_theta = np.zeros(num_neurons)
	mean_fires_out = 0
	for b in range(batch_size):
		data_idx = np.random.randint(len(train_data))
		times_nuer_fire = []
		membrane_volts = np.ones(num_neurons)*V_r
		for s in range(sim_steps):
			fired[:num_input] = convert_to_binary_1D_normalized(0.05*train_data[data_idx][0]/np.mean(train_data[data_idx][0]), input_prob) #this auto normalizes input strength (hopefully)
			tslfs, membrane_volts, neur_params, fired, g_E, g_I = update_net(tslfs, membrane_volts, neur_params, synapses, fired, g_E, g_I)
			real_jFs[:, s] = fired[:]
			if training:
				fired[num_input+num_hidden_exc+num_hidden_inhib:] = np.zeros(num_out)
				for n in range(num_out):
					if n%num_class == train_labels[data_idx]:
						if np.random.rand() < 1/5:
							fired[num_input+num_hidden_exc+num_hidden_inhib+n] = 1
			stdp_jFs[:, s] = fired[:]
			volts[:, s] = membrane_volts[:]
			
		sum_output_fires = np.zeros(num_class)
		for o in range(num_out):
			sum_output_fires[o%num_class] += np.sum(real_jFs[num_all - num_out + o, :])
			
		num_right += train_labels[data_idx] == np.argmax(sum_output_fires)	
		
		#print(np.mean(jFs[:num_input,:]), np.mean(jFs[num_input:num_input+num_hidden_exc,:]), np.mean(neur_params[:, 3]))	
		for n in range(num_all):
			times_nuer_fire.append(np.where(stdp_jFs[n, :] == 1)[0])
		
		if training:
			#print(np.mean(neur_params[:, 3]))
			#buffer_delta_theta += firing_rate_update(real_jFs, train_labels[data_idx])
			buffer_delta_syn_weights += stdp_but_faster(times_nuer_fire)

		mean_fires_out += np.mean(sum_output_fires)/batch_size

	
	#print(np.mean(synapses[:num_hidden_exc, :num_input]), np.mean(synapses[num_hidden_exc+num_hidden_inhib:, num_input:num_input+num_hidden_exc]), np.mean(synapses[:num_hidden_exc, num_input+num_hidden_exc:num_input+num_hidden_exc+num_hidden_inhib]))

	synapses += buffer_delta_syn_weights/batch_size
	#neur_params[:, 3] *= buffer_delta_theta/batch_size
	#neur_params[:, 3] = np.clip(neur_params[:, 3], 10, 30)
	synapses = np.clip(synapses, 0, 0.6)
	
	input_means = find_input_means()
	synapse_adjustments = beta / input_means
	synapses *= synapse_adjustments[:, np.newaxis]
	
	#synapses[:num_hidden_exc, :num_input] *= 0.15/np.mean(synapses[:num_hidden_exc, :num_input])
	#synapses[num_hidden_exc+num_hidden_inhib:, num_input:num_input+num_hidden_exc] *= 0.15/np.mean(synapses[num_hidden_exc+num_hidden_inhib:, num_input:num_input+num_hidden_exc])
	#synapses[:num_hidden_exc, num_input+num_hidden_exc:num_input+num_hidden_exc+num_hidden_inhib] *= 0.15/np.mean(synapses[:num_hidden_exc, num_input+num_hidden_exc:num_input+num_hidden_exc+num_hidden_inhib])
	#synapses = np.clip(synapses, 0, 0.6)
		
	version = "binary_train_synapses_" + str(e)
	np.save(version, synapses)
	version = "binary_train_neur_params_" + str(e)
	np.save(version, neur_params)

	diff = 0
	if e > 1:
		diff = np.mean(np.abs(synapses - np.load("binary_train_synapses_"+str(e-1)+".npy")))

	print(e, num_right/batch_size, mean_fires_out, np.mean(neur_params[:, 3]), np.mean(real_jFs[num_input:num_input+num_hidden_exc,:]), diff)

print(train_labels[data_idx])
print(np.sum(jFs[:num_input,:])/sim_steps)
print(np.sum(jFs[num_input:,:]))
print(np.sum(jFs[num_input+num_hidden_exc+num_hidden_inhib:,:]))


plt.figure()
plt.imshow(jFs)
plt.figure()
plt.imshow(volts)
plt.show()

