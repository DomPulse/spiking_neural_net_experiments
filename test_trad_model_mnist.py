#https://stackoverflow.com/questions/56435961/how-to-access-the-network-weights-while-using-pytorch-nn-sequential

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools



# dataloader arguments
batch_size = 128
data_path='D:\\Neuro Sci\\snntorch_learn_nn_robust\\mnist_data'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose([
			transforms.Resize((28, 28)),
			transforms.Grayscale(),
			transforms.ToTensor(),
			transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=False, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=False, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# neuron and simulation parameters

beta = 0.5
num_steps = 50
import torch
import torch.nn as nn
import torch.nn.functional as F

dropout = 0.15
fts_link = 32  # Increased number of filters
stt_link = 128  # Increased number of filters
out_features = 256  # Additional fully connected layer size

class Net(nn.Module):
	def __init__(self, p=dropout):
		super(Net, self).__init__()

		# Initialize convolutional layers, activation functions, and fully connected layers
		self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Reduced number of filters and kernel size
		self.relu1 = nn.ReLU()  # ReLU activation function for the first convolutional layer
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Reduced number of filters and kernel size
		self.relu2 = nn.ReLU()  # ReLU activation function for the second convolutional layer
		
		# Define dropout layer
		self.dropout = nn.Dropout(p)
		
		# Fully connected layers
		self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Reduced number of output features
		self.relu3 = nn.ReLU()  # ReLU activation function for the first fully connected layer
		self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes

	def forward(self, x):
		# Forward pass through the network
		
		# Convolution + ReLU + Max Pooling Layer 1
		x = self.conv1(x)  # Apply the first convolutional layer
		x = self.relu1(x)  # Apply the ReLU activation function
		x = F.max_pool2d(x, 2)  # Apply max pooling with a kernel size of 2

		# Convolution + ReLU + Max Pooling Layer 2
		x = self.conv2(x)  # Apply the second convolutional layer
		x = self.relu2(x)  # Apply the ReLU activation function
		x = F.max_pool2d(x, 2)  # Apply max pooling with a kernel size of 2

		# Flatten + Fully Connected Layers
		x = x.view(x.size(0), -1)  # Flatten the tensor from the convolutional layers
		x = self.fc1(x)  # Apply the first fully connected layer
		x = self.relu3(x)  # Apply the ReLU activation function
		x = self.dropout(x)  # Apply dropout
		x = self.fc2(x)  # Apply the output fully connected layer

		return x

# Initialize Network
net = Net(p=dropout).to(device)  # Move the network to the specified device (CPU or GPU)


def forward_pass(net, num_steps, data):
	output_rec = []  # List to record network outputs at each time step
	utils.reset(net)  # Reset the network (not necessary for standard networks, can be omitted or replaced if needed)

	# Forward pass through the network (no time steps in traditional NN, just a single pass)
	spk_out = net(data)  # Perform a forward pass through the network
	output_rec.append(spk_out)  # Record the output

	# Convert the recorded outputs to tensors
	return torch.stack(output_rec)

# already imported snntorch.functional as SF

def batch_accuracy(train_loader, net, num_steps):
	with torch.no_grad():
		total = 0
		acc = 0
		net.eval()

		train_loader = iter(train_loader)
	for data, targets in train_loader:
		data = data.to(device)
		targets = targets.to(device)
		spk_rec, _ = forward_pass(net, num_steps, data)

		acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
		total += spk_rec.size(1)
	return acc/total

num_epochs = 1
loss_hist = []
test_acc_hist = []
counter = 0
net.load_state_dict(torch.load("cnn_trad_model_mini.pth"))

conv1_shape = np.shape(net.conv1.weight.cpu().detach().numpy())
conv1_std = np.std(net.conv1.weight.cpu().detach().numpy())
#print(net.conv1.weight.cpu().detach().numpy())

conv2_shape = np.shape(net.conv2.weight.cpu().detach().numpy())
conv2_std = np.std(net.conv2.weight.cpu().detach().numpy())


#conv3_shape = np.shape(net.conv3.weight.cpu().detach().numpy())
#conv3_std = np.std(net.conv3.weight.cpu().detach().numpy())

#conv4_shape = np.shape(net.conv4.weight.cpu().detach().numpy())
#conv4_std = np.std(net.conv4.weight.cpu().detach().numpy())

fc1_shape = np.shape(net.fc1.weight.cpu().detach().numpy())
fc1_std = np.std(net.fc1.weight.cpu().detach().numpy())

fc2_shape = np.shape(net.fc2.weight.cpu().detach().numpy())
fc2_std = np.std(net.fc2.weight.cpu().detach().numpy())

#fc3_shape = np.shape(net.fc3.weight.cpu().detach().numpy())
#fc3_std = np.std(net.fc3.weight.cpu().detach().numpy())

correct = 0
total = 0

for i in range(1, 150):
	net.load_state_dict(torch.load("cnn_trad_model_mini.pth"))
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		for data in test_loader:
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)

			net.conv1.weight[:,:,:,:] = torch.from_numpy(np.add(net.conv1.weight.cpu().detach().numpy(), np.random.normal(0, conv1_std/i, conv1_shape))[:,:,:,:])
			net.conv2.weight[:,:,:,:] = torch.from_numpy(np.add(net.conv2.weight.cpu().detach().numpy(), np.random.normal(0, conv2_std/i, conv2_shape))[:,:,:,:])
			#net.conv3.weight[:,:,:,:] = torch.from_numpy(np.add(net.conv3.weight.cpu().detach().numpy(), np.random.normal(0, conv3_std/i, conv3_shape))[:,:,:,:])
			#net.conv4.weight[:,:,:,:] = torch.from_numpy(np.add(net.conv4.weight.cpu().detach().numpy(), np.random.normal(0, conv4_std/i, conv4_shape))[:,:,:,:])
			net.fc1.weight[:,:] = torch.from_numpy(np.add(net.fc1.weight.cpu().detach().numpy(), np.random.normal(0, fc1_std/i, fc1_shape))[:,:])
			net.fc2.weight[:,:] = torch.from_numpy(np.add(net.fc2.weight.cpu().detach().numpy(), np.random.normal(0, fc2_std/i, fc2_shape))[:,:])
			#net.fc3.weight[:,:] = torch.from_numpy(np.add(net.fc3.weight.cpu().detach().numpy(), np.random.normal(0, fc3_std/i, fc3_shape))[:,:])

			# calculate outputs by running images through the network
			outputs = net(inputs)
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		print(i, 100 * correct / total)



