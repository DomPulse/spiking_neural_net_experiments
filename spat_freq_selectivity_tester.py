import numpy as np
import matplotlib.pyplot as plt

epoch = 2
stren_mults = 10*np.exp(-1*np.linspace(0, 4, epoch))
print(stren_mults)
batch_size_prime = 10
batch_size_test = 100
learning_rate = 0.1

sim_length = 1000 #number of miliseconds in real time
del_t = 0.5 #in milliseconds
sim_steps = int(sim_length/del_t) #number of time steps taken

#all volts in mV
Tau = 100
V_r = -65 #resting
V_E = 0 #excitatory
V_I = -100 #inhibitory
V_thresh = -55 #threshold
theta_init = 0
V_reset = -65 #reset
t_refrac = 2

sqrt_num_input = 10
sqrt_num_out = 10
num_input = sqrt_num_input*sqrt_num_input
spat_freq = 2
test_stim_stren = 0.05
induce_stim_stren = 0.05
desired_fire_freq = 30
external_stim_freq = 100
freq_in_steps = int((1000/external_stim_freq)/del_t)
num_out = sqrt_num_out*sqrt_num_out
num_neurons = num_out
num_all = num_neurons + num_input
max_expected_fire = 30 #bit arbitrary init?
dropout = 0.2
frac_inhib = 0.0
weight_tune = 0.5

#these are, capacitence in nF, leak conductance in nano siemens, and the time constant for synaptic conductance which is currently unuse, then the threshold increase which is dynamic in this paper
excite_neur_params = [0.5, 25, 20, theta_init]
inhib_neur_params = [0.2, 20, 10, theta_init]

#these are the volatile neuron parameters, they are the voltage in mV and the time since the neuron last fired, initilized to 100 seconds as the neurons should start not firing
neur_vol = [-70, 100, 0]

exin_array = np.ones(num_all)
synapses = np.zeros((num_neurons, num_all))
defaults = np.zeros((num_neurons, num_all))
membrane_volts = np.ones(num_neurons)*V_r
g_E = np.zeros(num_neurons)
g_I = np.zeros(num_neurons)
fired = np.zeros(num_all)
tslfs = np.ones(num_all)*1000
neur_params = np.zeros((num_neurons, 4))
jFs = np.zeros((num_all, sim_steps))
volts = np.zeros((num_neurons, sim_steps))
look_back = 30
idx_pairs = []
#https://www.jneurosci.org/content/jneuro/2/1/32.full.pdf
#the good news ^

def simple_moving_average(data, window_size):
	"""
	Calculate the Simple Moving Average (SMA) of an array.
	
	Parameters:
		data (list or np.ndarray): The input array of numbers.
		window_size (int): The size of the moving window.
		
	Returns:
		np.ndarray: An array containing the SMA values. The result will be shorter by `window_size - 1` elements.
	"""
	if window_size <= 0:
		raise ValueError("Window size must be greater than 0")
	if window_size > len(data):
		raise ValueError("Window size cannot be larger than the data length")

	# Use NumPy's convolution for efficient calculation
	return np.convolve(data, np.ones(window_size) / window_size, mode='valid')*1000/(del_t) #should be firing rate Hz

def convert_to_binary_1D_normalized(array_2D, max_probability=0.2):
	flat_array = array_2D.flatten()
	probabilities = flat_array * max_probability
	binary_1D_array = np.random.rand(len(flat_array)) < probabilities
	return binary_1D_array.astype(int)  # Convert boolean to int (0 or 1)



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
		local_mem_volt[n] = V_reset*np.random.normal(1, 0.025)
		local_tslfs[num_input + n] = 0
		local_fired[num_input + n] = 1

	
	return local_tslfs, local_mem_volt, local_fired, local_g_E, local_g_I

exin_array = np.load('exin_array.npy')
neur_params = np.load('neur_params.npy')
synapses = np.load('synapses.npy')

'''
plt.imshow(synapses, aspect = 'auto', interpolation  = 'nearest', extent = [-num_input, num_neurons, num_neurons, 0])
plt.ylabel('post_syn_idx')
plt.xlabel('pre_syn_idx')
plt.show()
'''

#generates spatial frequency input patterns
woah_dude = np.zeros((2, sqrt_num_input, sqrt_num_input))
waver = np.sin(np.linspace(0, spat_freq*np.pi, sqrt_num_input))
woah_dude[0, :] = np.ones(sqrt_num_input)*waver[:]
for i in range(sqrt_num_input):
	woah_dude[1, :, i] = np.ones(sqrt_num_input)*waver[:]

#generates a mask that will effectively half the firing rate of the neurons we don't want firing for a given class
train_fire_mask = np.ones((2, num_all, sim_steps))
for n in range(num_neurons):
	train_fire_mask[n%2, num_input + n, ::2] = np.zeros(sim_steps//2)

avg_fire_rate = np.zeros((batch_size_prime, num_all))
above_avg_fire_by_class = np.zeros((2, num_out))
buffer_delta_syn_weights = np.zeros((num_neurons, num_all))
mean_response = []
i_have_stds = []
mean_syn = []
mean_in_region = np.zeros((2, batch_size_test))

num_right = 0
for e in range(epoch):
	if e == 0:
		this_batch_please = batch_size_prime
	elif e == 1:
		this_batch_please = batch_size_test
		avg_fire_rate = np.ones((batch_size_test, num_all))*np.mean(avg_fire_rate)
	for b in range(this_batch_please):
		train_class = np.random.randint(0, 2)
		#train_class = 0
		smoothed_fires = np.zeros((num_all, sim_steps-look_back+1))
		train_smoothed_fires = np.zeros((num_all, sim_steps-look_back+1))
		jFs = np.zeros((num_all, sim_steps))
		membrane_volts = np.random.normal(1, 0.025, num_neurons)*V_r
		g_E = np.zeros(num_neurons)
		g_I = np.zeros(num_neurons)
		tslfs = np.ones(num_all)*1000
	
		for s in range(sim_steps):
			if b > 10 and b < 20 and e > 0:
				fired[:num_input] = convert_to_binary_1D_normalized(woah_dude[train_class], induce_stim_stren)
			else:
				fired[:num_input] = convert_to_binary_1D_normalized(woah_dude[train_class], test_stim_stren)
			tslfs, membrane_volts, fired, g_E, g_I = update_net(tslfs, membrane_volts, neur_params, synapses, fired, g_E, g_I)
			jFs[:, s] = fired[:]
			volts[:, s] = membrane_volts[:]
		
		avg_fire_rate[b] = np.sum(jFs, axis = 1)*1000/(sim_length) #average for each neuron in this run

		if e > 0:
			c_avg = np.mean(avg_fire_rate, axis = 0) #average for each neuron over the last batch size (like 100 or something)
			for n in range(num_all):
				smoothed_fires[n, :] = simple_moving_average(jFs[n], look_back)
		
			above_avg_fire = avg_fire_rate[b, num_all - num_out:] >= np.mean(avg_fire_rate[b, num_all - num_out:])
			above_avg_fire_by_class[train_class] += above_avg_fire
			
			mean_response.append(np.mean(smoothed_fires[num_input:]))
			mean_in_region[0, b] = np.mean(smoothed_fires[num_input::2])
			mean_in_region[1, b] = np.mean(smoothed_fires[num_input+1::2])
			print(b, train_class, mean_in_region[0, b] > mean_in_region[1, b])
			num_right += train_class == (mean_in_region[0, b] > mean_in_region[1, b] )
			#plt.imshow(smoothed_fires, aspect = 'auto', interpolation  = 'nearest')
			#plt.show()

print(num_right)

plt.figure()
plt.title('mean response')
plt.plot(mean_in_region[0], label = 'region 1')
plt.plot(mean_in_region[1], label = 'region 2')
plt.legend()

plt.figure()
plt.imshow(synapses, aspect = 'auto', interpolation  = 'nearest', extent = [-num_input, num_neurons, num_neurons, 0])
plt.ylabel('post_syn_idx')
plt.xlabel('pre_syn_idx')

plt.show()

