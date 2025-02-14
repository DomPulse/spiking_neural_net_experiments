import numpy as np
import matplotlib.pyplot as plt
#taken from https://sci-hub.se/https://doi.org/10.1023/a:1008857906763

C0 = 1 #1milliMolar
labda = 1.61 #cm
start_rad = 3 #cm
v = 1/40 #this is the starting speed in cm/sec
half_length = 5 #how far are the walls from the center? consult this variable for that information in cm
time_step = 0.1 #time step in seconds
sim_length = 5*60
sim_steps = int(sim_length/time_step)
num_neurons = 5
num_worms = 150
batch_size = 100
epochs = 100

def init_syns(num_neurons, chemo_sens_idxs = [0], mult = [1, 1, 1], over_all_stren = 1):
	temp_As = np.random.rand(num_neurons, num_neurons)*over_all_stren*mult[0] #matrix weight
	temp_bs = np.random.rand(num_neurons)*over_all_stren*mult[1]  #b is for bias
	temp_cs = np.zeros(num_neurons) #selects which cells are sensitive to chemo-sense
	temp_ks = np.random.rand(num_neurons)*over_all_stren*mult[2] #strength of chemo-sense
	for pre_syn_idx in range(num_neurons):
		if pre_syn_idx in chemo_sens_idxs:
			temp_cs[pre_syn_idx] = 1
		for post_syn_idx in range(num_neurons):
			if post_syn_idx == pre_syn_idx:
				temp_As[post_syn_idx, pre_syn_idx] = 0
	return temp_As, temp_bs, temp_cs, temp_ks

def i_maka_da_gradient(dim = 101):
	xs = np.linspace(-half_length, half_length, dim)
	ys = np.linspace(-half_length, half_length, dim)
	temp_grad = np.zeros((dim, dim))
	for i in range(dim):
		for j in range(dim):
			temp_grad[j, i] = C0*np.exp(-(xs[i]**2 + ys[j]**2)/(2*labda**2))
	return temp_grad, xs, ys

def update_voltage(Vs, As, bs, cs, ks, C, dt):
	dV = np.matmul(As, Vs) + bs + cs*ks*C
	Vs += dV*dt
	return Vs, dV

def death_and_sex(start_As, start_bs, start_cs, start_ks, scores):
	score_to_beat = (np.max(scores)+np.mean(scores))/2
	idxs_to_keep = np.where(scores >= score_to_beat)
	all_As = []
	all_bs = []
	all_cs = []
	all_ks = []
	for w in range(num_worms):
		mommy_idx = np.random.choice(idxs_to_keep[0])
		daddy_idx = np.random.choice(idxs_to_keep[0])
		As, bs, cs, ks = init_syns(num_neurons, mult = (np.random.rand(3)-0.5)/100, over_all_stren = 1/100)
		all_As.append((start_As[mommy_idx] + start_As[daddy_idx])/2 + As)
		all_bs.append((start_bs[mommy_idx] + start_bs[daddy_idx])/2 + bs)
		all_cs.append((start_cs[mommy_idx] + start_cs[daddy_idx])/2 + cs)
		all_ks.append((start_ks[mommy_idx] + start_ks[daddy_idx])/2 + ks)

	all_As = np.asarray(all_As)
	all_bs = np.asarray(all_bs)
	all_cs = np.asarray(all_cs)
	all_ks = np.asarray(all_ks)

	return all_As, all_bs, all_cs, all_ks

#yes i am sure there is a more efficient way for this, i just dont care
gamma = np.random.rand() #this essentially determines the strength of the turning speed
all_gammas = np.random.rand(num_worms)
all_As = []
all_bs = []
all_cs = []
all_ks = []
for w in range(num_worms):
	As, bs, cs, ks = init_syns(num_neurons, mult = [-0.003,  0.03,  0.12], over_all_stren = 1/10)
	all_As.append(As)
	all_bs.append(bs)
	all_cs.append(cs)
	all_ks.append(ks)

all_As = np.asarray(all_As)
all_bs = np.asarray(all_bs)
all_cs = np.asarray(all_cs)
all_ks = np.asarray(all_ks)

gradient, xs, ys = i_maka_da_gradient()
for e in range(epochs):
	scores = np.zeros(num_worms)
	for b in range(batch_size):
		start_phi = np.random.rand()*np.pi*2 #radians, position relative to center NOT WHERE THE WORM IS FACING
		start_theta = np.random.rand()*np.pi*2 #starting angle the worm is facing
		for w in range(num_worms):
			Vs = -65*np.ones(num_neurons) #initial voltages, idk what they should be
			pos = np.asarray([start_rad*np.sin(start_phi), start_rad*np.cos(start_phi)])
			theta = start_theta
			As, bs, cs, ks = all_As[w], all_bs[w], all_cs[w], all_ks[w]
			for s in range(sim_steps):
				i = np.argmin(np.abs(pos[0] - xs))
				j = np.argmin(np.abs(pos[1] - ys))
				C = gradient[j, i]
				Vs, dVs = update_voltage(Vs, As, bs, cs, ks, C, time_step)
				theta += all_gammas[w]*(Vs[-1] - Vs[-2])
				pos[0] += v*np.cos(theta)
				pos[1] += v*np.sin(theta)
				pos = np.clip(pos, -half_length, half_length)
			dist = np.sqrt(np.sum(pos**2))
			scores[w] += (3 - dist)/batch_size
	print(e, np.max(scores), np.argmax(scores)) 
	all_As, all_bs, all_cs, all_ks = death_and_sex(all_As, all_bs, all_cs, all_ks, scores)
	np.save('all_As_'+str(e), all_As)
	np.save('all_bs_'+str(e), all_bs)
	np.save('all_cs_'+str(e), all_cs)
	np.save('all_ks_'+str(e), all_ks)
