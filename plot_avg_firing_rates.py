import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

num_trials = 100
num_input = 784
num_in_region = 1150
num_regions = 2
num_inhib = 700

# Load CSV file into a pandas DataFrame
csv_file_fires = 'D:\\Neuro Sci\\bi_stable_competition\\mnist_input\\mnist_input\\fires.csv'  # Replace with your CSV filename
#csv_file_volts = 'D:\\Neuro Sci\\bi_stable_competition\\mnist_input\\mnist_input\\volts.csv'  # Replace with your CSV filename
#csv_file_fires = "D:\\Neuro Sci\\learn_cpp\\lif_bistable\\lif_bistable\\fires.csv"
#csv_file_volts = "D:\\Neuro Sci\\learn_cpp\\lif_bistable\\lif_bistable\\volts.csv"
data = pd.read_csv(csv_file_fires, header=None)  # Use header=None if there's no header row

# Convert the DataFrame to a numpy array (optional, but useful for customization)
data_array = data.values
data_array = np.asarray(data_array, dtype = float)
region_one_avg = np.mean(data_array[len(data_array)-200:, num_input:num_input+num_in_region])
region_two_avg = np.mean(data_array[len(data_array)-200:, num_input+num_in_region:num_input+num_regions*num_in_region])
print("{:.3f}".format(np.abs(region_two_avg- region_one_avg)))

sim_steps = 999
times = np.linspace(0,sim_steps,sim_steps)
all_fires_win = np.zeros((sim_steps, num_trials))
all_fires_loss = np.zeros((sim_steps, num_trials))

avg_fires_win = np.zeros(sim_steps)
avg_fires_loss = np.zeros(sim_steps)
num_flip = 0
num_region_one = 0
for trial in range(num_trials):
	print(trial)

	peak_one_avg = np.mean(data_array[sim_steps*trial + 80:sim_steps*trial + 125, num_input:num_input+num_in_region])
	peak_two_avg = np.mean(data_array[sim_steps*trial + 80:sim_steps*trial + 125, num_input+num_in_region:num_input+num_regions*num_in_region])

	region_one_avg = np.mean(data_array[sim_steps*trial + sim_steps-200:sim_steps*trial + sim_steps, num_input:num_input+num_in_region])
	region_two_avg = np.mean(data_array[sim_steps*trial + sim_steps-200:sim_steps*trial + sim_steps, num_input+num_in_region:num_input+num_regions*num_in_region])
	if region_one_avg > region_two_avg:
		#print('gooning')
		num_region_one += 1
		if peak_one_avg < peak_two_avg:
			num_flip += 1
		for t in range(sim_steps):
			all_fires_win[t][trial] = np.mean(data_array[sim_steps*trial + t, num_input:num_input+num_in_region])
			all_fires_loss[t][trial] = np.mean(data_array[sim_steps*trial + t, num_input+num_in_region:num_input+num_regions*num_in_region])
	else:
		if peak_one_avg > peak_two_avg:
			num_flip += 1
		for t in range(sim_steps):
			all_fires_loss[t][trial] = np.mean(data_array[sim_steps*trial + t, num_input:num_input+num_in_region])
			all_fires_win[t][trial] = np.mean(data_array[sim_steps*trial + t, num_input+num_in_region:num_input+num_regions*num_in_region])

print(num_flip)
print(num_flip/num_trials)
print(num_region_one/num_trials)

avg_fires_win = np.mean(all_fires_win, axis = 1)
std_win = np.std(all_fires_win, axis = 1)
avg_fires_loss = np.mean(all_fires_loss, axis = 1)
std_loss = np.std(all_fires_loss, axis = 1)
plt.plot(times, avg_fires_win, color = 'blue', label = "winner firing rate")
plt.plot(times, avg_fires_loss, color = 'orange', label = "loser firing rate")
plt.fill_between(times, avg_fires_win - std_win, avg_fires_win + std_win, color='blue', alpha=0.3)
plt.fill_between(times, avg_fires_loss - std_loss, avg_fires_loss + std_loss, color='orange', alpha=0.3)
plt.legend(loc = 'upper left')
plt.show()

