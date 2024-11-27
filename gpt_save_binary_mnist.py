import torch
from torchvision import datasets, transforms
import numpy as np

# Set up the data path
data_path = 'D:\\Neuro Sci\\snntorch_learn_nn_robust\\mnist_data'

# Define the transformation for MNIST data
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

# Load MNIST train and test datasets
mnist_train = datasets.MNIST(data_path, train=True, download=False, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=False, transform=transform)

# Filter function to get only ones and zeros
def filter_ones_zeros(dataset):
    filtered_data = []
    filtered_labels = []
    for img, label in dataset:
        #if label in [0, 1]:
        filtered_data.append(img.numpy())
        filtered_labels.append(label)
    
    # Convert lists to NumPy arrays
    filtered_data = np.array(filtered_data)
    filtered_labels = np.array(filtered_labels)
    
    return filtered_data, filtered_labels

# Filter train and test data
train_data, train_labels = filter_ones_zeros(mnist_train)
test_data, test_labels = filter_ones_zeros(mnist_test)

# Save the filtered data and labels to NumPy arrays
np.save('mnist_train_data.npy', train_data)
np.save('mnist_train_labels.npy', train_labels)
np.save('mnist_test_data.npy', test_data)
np.save('mnist_test_labels.npy', test_labels)

print("Data saved successfully!")
