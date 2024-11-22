import numpy as np
import matplotlib.pyplot as plt

synapses_1 = np.load("synapses_10.npy")
synapses = np.load("lateral_inhib_first_test\\synapses_100.npy")
#synapses_2 = np.load("synapses_100.npy")

#neur_params = np.load("neur_params_100.npy")

#print(neur_params[0])

plt.figure()
plt.imshow(synapses_1[:400,:784])
plt.show()

for n in range(200, 400):
	plt.imshow(synapses[n,:784].reshape((28,28)))
	plt.show()

#plt.figure()
#plt.imshow(synapses_1)
#plt.figure()
#plt.imshow(synapses_2 - synapses_1)
