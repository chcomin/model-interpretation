import random
import numpy as np
import torch
import torchvision
from sklearn.decomposition import PCA

MNIST_MEAN = 33.3
MNIST_STD = 78.6

def transform(img):
    """Transformation for the MNIST dataset. PIL->numpy->z-score->tensor."""
    img = np.array(img, dtype=np.float32)
    img = normalize(img, MNIST_MEAN, MNIST_STD)
    return torch.from_numpy(img)

def normalize(data, mean=None, std=None):
    """Z-score for an array. Note that all features are normalized by the same
    mean and standard deviation."""

    if mean is None: mean = data.mean()
    if std is None: std = data.std()
    
    return (data-mean)/std

def get_mnist(root, n):
    """Load the MNIST dataset and normalize it."""

    ds_train = torchvision.datasets.MNIST(root, train=True, transform=transform, download=False)
    ds_valid = torchvision.datasets.MNIST(root, train=False, transform=transform, download=False)

    # Extract random subset for training data
    indices = random.sample(range(len(ds_train)), n)
    ds_train.data = ds_train.data[indices]
    ds_train.targets = ds_train.targets[indices]

    return ds_train, ds_valid

def get_mnist_numpy(root, n):
    """Load the MNIST dataset as a numpy array. For instance, the train dataset 
    is loaded as a single matrix, where each row is an image."""

    ds_train, ds_valid = get_mnist(root, n)
    data_train = np.array(ds_train.data.reshape(n, -1), dtype=np.float32)
    labels_train = np.array(ds_train.targets)
    data_valid = np.array(ds_valid.data.reshape(len(ds_valid), -1), dtype=np.float32)
    labels_valid = np.array(ds_valid.targets)

    data_train = normalize(data_train, MNIST_MEAN, MNIST_STD)
    data_valid = normalize(data_valid, MNIST_MEAN, MNIST_STD)

    return data_train, labels_train, data_valid, labels_valid

def reduce_dimensionality(data_train, data_valid, var_keep=0.95):
    """Apply PCA to the data."""

    mapper = PCA(n_components=var_keep, whiten=False)
    data_train_pca = mapper.fit_transform(data_train)
    data_valid_pca = mapper.transform(data_valid)
    u = data_train_pca.mean()
    s = data_train_pca.std()
    data_train_pca = normalize(data_train_pca, u, s)
    data_valid_pca = normalize(data_valid_pca, u, s)

    return data_train_pca, data_valid_pca
