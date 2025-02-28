import numpy as np
import matplotlib.pyplot as plt


epoch = 2
stren_mults = 10*np.exp(-1*np.linspace(0, 4, epoch))
print(stren_mults)
batch_size_prime = 50
batch_size_test = 5000
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
sqrt_num_out = 6
num_input = sqrt_num_input*sqrt_num_input
spat_freq = 2
test_stim_stren = 0.05
induce_stim_stren = 0.05
desired_fire_freq = 30
external_stim_freq = 100
freq_in_steps = int((1000/external_stim_freq)/del_t)
num_out = sqrt_num_out*sqrt_num_out
num_hid = 75
num_hid_layer = 2
num_neurons = num_out + num_hid*num_hid_layer
num_all = num_neurons + num_input
max_expected_fire = 30 #bit arbitrary init?
dropout = 0
weight_tune = 0.7
weight_std = 0.2

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

def phi(c, c_avg, c0, p = 2, strength = 1):
	#https://www.desmos.com/calculator/wriw5fmx6b
	#that desmos graph has the template function
	#strength is a new addition, it is basically learning rate
	thresh = c_avg*(c_avg/c0)**p
	return c*(c-thresh)/(1+c**2)

def cherry_pie(c, c_avg, c0):
	del_synapses = np.zeros((num_neurons, num_all))
	to_deriv = np.linspace(0, 100, 200)
	zero_idx = np.argmin(np.abs(to_deriv - c0))
	slope = (phi(to_deriv[zero_idx+1], c0, c0) - phi(to_deriv[zero_idx-1], c0, c0))/(to_deriv[zero_idx+1] - to_deriv[zero_idx-1]) #we find tangent near the c0 point so we can set the decay to cancel small canges i guess
	slope = 0.03
	norm_phi = np.mean(c)*200 #ngl idk where the factor comes from but it is aparently needed
	norm_inhib = np.mean(c)*2000
	ums = []
	dums = []
	for pair in idx_pairs:
		pre_syn_idx = pair[0]
		post_syn_idx = pair[1]

		#um = slope*(synapses[post_syn_idx][pre_syn_idx] - defaults[post_syn_idx][pre_syn_idx])
		um = slope*(np.mean(synapses[post_syn_idx]) - defaults[post_syn_idx][pre_syn_idx]) #this is inspired by https://www.nature.com/articles/nn1100_1178 and should increase competition
		del_synapses[post_syn_idx][pre_syn_idx] -= um
		ums.append(um)

		#we be trying something
		#diff = (synapses[post_syn_idx][pre_syn_idx] - defaults[post_syn_idx][pre_syn_idx])
		#del_synapses[post_syn_idx][pre_syn_idx] -= slope*((diff) + 0.1*diff**3)
		
		#this is also inspired by the https://www.nature.com/articles/nn1100_1178 and um should be closer to the stdp based inhibitory neuron rule i guess
		if exin_array[pre_syn_idx] == 1:
			d = c[pre_syn_idx]
			dum = np.mean(phi(c[post_syn_idx + num_input], c_avg[post_syn_idx + num_input], c0)*d)/norm_phi #div by mean c to normalize or sm
		else:
			dum = np.mean(c[post_syn_idx + num_input])/(norm_inhib)
			dums.append(dum)
		del_synapses[post_syn_idx][pre_syn_idx] += dum
		

	#print('dummy', np.mean(dums), np.mean(ums))

	del_synapses *= learning_rate
	return del_synapses

def convert_to_binary_1D_normalized(array_2D, max_probability=0.2):
	flat_array = array_2D.flatten()
	probabilities = flat_array * max_probability
	binary_1D_array = np.random.rand(len(flat_array)) < probabilities
	return binary_1D_array.astype(int)  # Convert boolean to int (0 or 1)

#this do be the architecture generation
for pre_syn_idx in range(num_all):
	adjusted_pre_syn_idx = pre_syn_idx - num_input
	if adjusted_pre_syn_idx < 0:
		adjusted_pre_syn_idx = num_all + 1000
	for post_syn_idx in range(num_neurons):
		if np.random.rand() < dropout:
			pass

		elif adjusted_pre_syn_idx >= num_hid*(num_hid_layer - 1) and adjusted_pre_syn_idx < num_hid*num_hid_layer and post_syn_idx >= num_hid*num_hid_layer: #lastt hid to out
			synapses[post_syn_idx, pre_syn_idx] = np.max([np.random.normal(weight_tune, weight_std), 0])
			defaults[post_syn_idx, pre_syn_idx] = weight_tune
			idx_pairs.append([pre_syn_idx, post_syn_idx])

		elif pre_syn_idx < num_input and post_syn_idx < num_hid: #in to first hid
			synapses[post_syn_idx, pre_syn_idx] = np.max([np.random.normal(weight_tune, weight_std), 0])
			defaults[post_syn_idx, pre_syn_idx] = weight_tune
			idx_pairs.append([pre_syn_idx, post_syn_idx])

		elif post_syn_idx >= num_hid*(num_hid_layer - 1) and post_syn_idx < num_hid*num_hid_layer and adjusted_pre_syn_idx >= num_hid*num_hid_layer and adjusted_pre_syn_idx < num_neurons: #out to last hid
			synapses[post_syn_idx, pre_syn_idx] = np.max([np.random.normal(weight_tune, weight_std), 0])
			defaults[post_syn_idx, pre_syn_idx] = weight_tune
			idx_pairs.append([pre_syn_idx, post_syn_idx])

for i in range(num_hid_layer):
	for kinda_post_syn_idx in range(num_hid):
		#gets the hid to same hid
		for kinda_pre_syn_idx in range(num_hid):
			if np.random.rand() < dropout:
				pass
			else:
				post_syn_idx = kinda_post_syn_idx + num_hid*i
				pre_syn_idx = num_input + kinda_pre_syn_idx + num_hid*i
				synapses[post_syn_idx, pre_syn_idx] = np.max([np.random.normal(weight_tune, weight_std), 0])
				defaults[post_syn_idx, pre_syn_idx] = weight_tune
				idx_pairs.append([pre_syn_idx, post_syn_idx])

		#hid to next hidden layer
		if i < num_hid_layer - 1:
			for kinda_pre_syn_idx in range(num_hid):
				if np.random.rand() < dropout:
					pass
				else:
					post_syn_idx = kinda_post_syn_idx + num_hid*(i+1)
					pre_syn_idx = num_input + kinda_pre_syn_idx + num_hid*i
					synapses[post_syn_idx, pre_syn_idx] = np.max([np.random.normal(weight_tune, weight_std), 0])
					defaults[post_syn_idx, pre_syn_idx] = weight_tune
					idx_pairs.append([pre_syn_idx, post_syn_idx])

for n in range(num_neurons):
	neur_params[n, :] = excite_neur_params[:]
	if n < num_hid and np.random.rand() > 0.5:
		neur_params[n, :] = inhib_neur_params[:]
		exin_array[num_input + n] = 0
	
idx_pairs = np.array(idx_pairs)
print(len(idx_pairs))

plt.imshow(synapses, aspect = 'auto', interpolation  = 'nearest', extent = [-num_input, num_neurons, num_neurons, 0])
plt.ylabel('post_syn_idx')
plt.xlabel('pre_syn_idx')
plt.show()

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

np.save('exin_array', exin_array)
np.save('neur_params', neur_params)


#generates spatial frequency input patterns
xs = np.linspace(-1, 1, sqrt_num_input)
ys = np.linspace(-1, 1, sqrt_num_input)
gaming = np.zeros((sqrt_num_input, sqrt_num_input))
angles = [0, np.pi/2]
offset = 0

#generates a mask that will effectively half the firing rate of the neurons we don't want firing for a given class
train_fire_mask = np.ones((2, num_all, sim_steps))
for n in range(num_hid, num_neurons):
	train_fire_mask[n%2, num_input + n, ::2] = np.zeros(sim_steps//2)

avg_fire_rate = np.zeros((batch_size_prime, num_all))
above_avg_fire_by_class = np.zeros((2, num_out))
buffer_delta_syn_weights = np.zeros((num_neurons, num_all))
mean_response = []
i_have_stds = []
mean_syn = []
mean_in_region = np.zeros((2, batch_size_test))

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

		theta = angles[train_class]
		offset = np.random.rand()*0
		for i in range(sqrt_num_input):
			for j in range(sqrt_num_input):
				r = np.sqrt((xs[i]**2) + (ys[j]**2))
				alpha = np.arctan2(xs[i], ys[j])
				d = r*np.sin(alpha - theta)
				gaming[i, j] = np.sin(spat_freq*np.pi*(d + offset))
		#gaming -= np.min(gaming)
		#gaming /= 4

		for s in range(sim_steps):
			fired[:num_input] = convert_to_binary_1D_normalized(gaming, test_stim_stren)
			tslfs, membrane_volts, fired, g_E, g_I = update_net(tslfs, membrane_volts, neur_params, synapses, fired, g_E, g_I)
			jFs[:, s] = fired[:]
			volts[:, s] = membrane_volts[:]
		
		train_jFs = jFs*train_fire_mask[train_class]
		
		'''
		if e != 0:
			#plt.imshow(synapses, aspect = 'auto', interpolation  = 'nearest')
			#plt.show()
			plt.imshow(jFs, aspect = 'auto', interpolation  = 'nearest')
			plt.show()
		'''
		
		avg_fire_rate[b] = np.sum(jFs, axis = 1)*1000/(sim_length) #average for each neuron in this run

		if e > 0:
			c_avg = np.mean(avg_fire_rate, axis = 0) #average for each neuron over the last batch size (like 100 or something)
			for n in range(num_all):
				train_smoothed_fires[n, :] = simple_moving_average(train_jFs[n], look_back)
				smoothed_fires[n, :] = simple_moving_average(jFs[n], look_back)
			synapses += cherry_pie(train_smoothed_fires, c_avg, np.mean(c_avg))
			synapses = np.clip(synapses, 0, 3*weight_tune)
		
			if b%20 == 0:
				print(b, np.mean(smoothed_fires[num_input:]), np.mean(c_avg[num_input:]), np.mean(synapses))
				#plt.imshow(synapses, aspect = 'auto', interpolation  = 'nearest', extent = [-num_input, num_neurons, num_neurons, 0])
				#plt.ylabel('post_syn_idx')
				#plt.xlabel('pre_syn_idx')
				#plt.show()
			
			i_have_stds.append(np.std(synapses))
			mean_syn.append(np.mean(synapses))
			mean_in_region[0, b] = np.mean(smoothed_fires[num_input+num_hid::2])
			mean_in_region[1, b] = np.mean(smoothed_fires[num_input+num_hid+1::2])
			#plt.imshow(smoothed_fires, aspect = 'auto', interpolation  = 'nearest')
			#plt.show()

np.save('synapses', synapses)

plt.figure()
plt.title('mean response')
plt.plot((mean_in_region[0]+mean_in_region[1])/2, label = 'mean for both')
plt.plot(mean_in_region[0], label = 'region 1')
plt.plot(mean_in_region[1], label = 'region 2')
plt.legend()

plt.figure()
plt.title('synapse avg')
plt.plot(mean_syn)

plt.figure()
plt.title('synapse std')
plt.plot(i_have_stds)

plt.figure()
plt.imshow(synapses, aspect = 'auto', interpolation  = 'nearest', extent = [-num_input, num_neurons, num_neurons, 0])
plt.ylabel('post_syn_idx')
plt.xlabel('pre_syn_idx')

plt.show()

