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

net.load_state_dict(torch.load("bigger_snn_model.pth"))

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

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


num_epochs = 1
loss_hist = []
test_acc_hist = []
counter = 0

print(net)
print(net[0].weight[0][0][0][0].item())
print(net[3].weight[0][0][0][0].item())
print(net[7].weight[0][0].item())
print(net[7].weight.cpu().detach().numpy())
girl_boss_gaming = net[7].weight.cpu().detach().numpy()
center = np.mean(girl_boss_gaming)
std = np.std(girl_boss_gaming)/5
shape = (np.shape(girl_boss_gaming))
noise = np.random.normal(center, std, shape)
girl_boss_gaming = np.add(girl_boss_gaming, noise)
#plt.hist(girl_boss_gaming.flatten())
#plt.show()

virgin_first_layer = net[0].weight.cpu().detach().numpy()
virgin_second_layer = net[3].weight.cpu().detach().numpy()
virgin_third_layer = net[7].weight.cpu().detach().numpy()
virgin_fourth_layer = net[9].weight.cpu().detach().numpy()

first_shape = (np.shape(virgin_first_layer))
second_shape = (np.shape(virgin_second_layer))
third_shape = (np.shape(virgin_third_layer))
fourth_shape = (np.shape(virgin_fourth_layer))

first_std = np.std(virgin_first_layer)
second_std = np.std(virgin_second_layer)
thrid_std = np.std(virgin_third_layer)
fourth_std = (np.std(virgin_fourth_layer))

for i in range(1, 6):
	with torch.no_grad():
		net[0].weight[:,:,:,:] = torch.from_numpy(virgin_first_layer[:,:,:,:])
		net[3].weight[:,:,:,:] = torch.from_numpy(virgin_second_layer[:,:,:,:])
		net[7].weight[:,:] = torch.from_numpy(virgin_third_layer[:, :])
		net[9].weight[:,:] = torch.from_numpy(virgin_fourth_layer[:, :])
		net.eval()
		test_acc_virgin = batch_accuracy(test_loader, net, num_steps)
		noise_1 = np.random.normal(0, first_std/i, first_shape)
		noise_2 = np.random.normal(0, second_std/i, second_shape)
		noise_3 = np.random.normal(0, thrid_std/i, third_shape)
		noise_4 = np.random.normal(0, fourth_std/i, fourth_shape)

		net[0].weight[:,:,:,:] = torch.from_numpy(np.add(virgin_first_layer, noise_1)[:,:,:,:])
		net[3].weight[:,:,:,:] = torch.from_numpy(np.add(virgin_second_layer, noise_2)[:,:,:,:])
		net[7].weight[:,:] = torch.from_numpy(np.add(virgin_third_layer, noise_3)[:,:])
		net[9].weight[:,:] = torch.from_numpy(np.add(virgin_fourth_layer, noise_4)[:,:])
		net.eval()
		test_acc = batch_accuracy(test_loader, net, num_steps)
		print(test_acc_virgin, test_acc, i)
		


