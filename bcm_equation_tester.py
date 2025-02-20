import numpy as np
import matplotlib.pyplot as plt
#https://www.jneurosci.org/content/jneuro/2/1/32.full.pdf
#the good news ^

def phi(c, c_avg, c0, p = 0.75, strength = 1):
	#https://www.desmos.com/calculator/izy6an5owp
	#that desmos graph has the template function
	#strength is a new addition, it is basically learning rate
	#I changed this function from the desmos one a little cuz c0 is supposed to be it's own fixed thing, not dependant on c_avg
	thresh = c_avg*(c_avg/c0)**p
	return c*(c-thresh)/(1+c**2)

c = np.linspace(0, 0.08, 100)
plt.plot(c, phi(c, 0.04, 0.025))
plt.show()

