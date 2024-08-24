#https://stackoverflow.com/questions/56435961/how-to-access-the-network-weights-while-using-pytorch-nn-sequential

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5

lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

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

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50

fts_link = 32  # Increased number of filters
stt_link = 32  # Increased number of filters
out_features = 32  # Additional fully connected layer size

# Define Network
class Net(nn.Module):
	def __init__(self):
		super().__init__()

		# Initialize layers
		self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
		self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
		
		self.fc1 = nn.Linear(32 * 7 * 7, 128)
		self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
		self.fc2 = nn.Linear(128, 10)
		self.lif6 = snn.Leaky(beta=beta, spike_grad=spike_grad)



net = nn.Sequential(
	nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
	nn.MaxPool2d(2),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
	nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
	nn.MaxPool2d(2),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

	nn.Flatten(),

	nn.Linear(32 * 7 * 7, 128),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

	nn.Linear(128, 10),
	snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
	).to(device)


def forward_pass(net, num_steps, data):
	mem_rec = []
	spk_rec = []
	utils.reset(net)	# resets hidden states for all LIF neurons in net

	for step in range(num_steps):
		spk_out, mem_out = net(data)
		spk_rec.append(spk_out)
		mem_rec.append(mem_out)

	return torch.stack(spk_rec), torch.stack(mem_rec)

# already imported snntorch.functional as SF
loss_fn = SF.ce_rate_loss()

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


optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 1
loss_hist = []
test_acc_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):

	# Training loop
	for data, targets in iter(train_loader):
		data = data.to(device)
		targets = targets.to(device)

		# forward pass
		net.train()
		spk_rec, _ = forward_pass(net, num_steps, data)

		# initialize the loss & sum over time
		loss_val = loss_fn(spk_rec, targets)

		# Gradient calculation + weight update
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		# Store loss history for future plotting
		loss_hist.append(loss_val.item())

		# Test set
		if counter % 50 == 0:
			with torch.no_grad():
				net.eval()

				# Test set forward pass
				test_acc = batch_accuracy(test_loader, net, num_steps)
				print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
				test_acc_hist.append(test_acc.item())

		counter += 1

# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(test_acc_hist)
plt.title("Test Set Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

torch.save(net.state_dict(), "bigger_snn_model.pth")