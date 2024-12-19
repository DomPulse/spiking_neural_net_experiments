import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Step 1: Create a list of n x n arrays (frames for the animation)
frames = []

def gaming(syns):
    silly_billy = np.zeros((20*28, 20*28))
    for i in range(20):
        for j in range(20):
            um = i*20 + j
            silly_billy[i*28:(i+1)*28, j*28:(j+1)*28] = syns[um,:784].reshape((28,28))[:,:]
    return silly_billy

for idx in range(5, 280, 5):
    name = "quad_train_synapses_" + str(idx)+".npy"
    frames.append(gaming(np.load(name)))

# Step 2: Set up the figure and axis
fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap='viridis', interpolation='none')

# Step 3: Define the update function
def update(frame):
    im.set_array(frame)
    return [im]

# Step 4: Create the animation
ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=200)

# Step 5: Save or show the animation
plt.show()