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

num_sims = 1
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


syn_weights = np.load("syn_weights.npy")
in_to_excite = np.load("in_to_excite.npy")

#syn_weights[0:num_excite,num_excite:num_neurons] = np.zeros((num_excite, num_inhib))
#syn_weights[num_excite:num_neurons,0:num_excite] = np.zeros((num_inhib, num_excite))

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

print("im in my goon cave")

decision_step = 270
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
		plt.figure()
		plt.hist(np.mean(jFs[num_inputs + num_in_region*j:num_inputs + num_in_region*(j+1)], axis = 1))

	print(firing_rates, firing_rates[train_labels[data_idx]] == np.max(firing_rates))
	


	plt.figure()
	plt.imshow(jFs)
	plt.show()




