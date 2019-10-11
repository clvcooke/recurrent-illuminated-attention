import numpy as np
import os
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        image = self.data_x[index]
        label = self.data_y[index]

        return (image, label)


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
      In the paper, this number is set to 0.1.
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    task = 'MNIST'
    if task == 'malaria':
        datapath_train_images = r"C:\Users\clvco\Documents\Code\K-space_rl-data\malaria_norm.npy"
        datapath_train_labels = r"C:\Users\clvco\Documents\Code\K-space_rl-data\malaria_labels.npy"

        datapath_test_images = None
        datapath_test_labels = None
        train_fact = 255
        shuffle_seed = 3
        channels = 96
        classes = 2
    elif task == 'fashion':
        datapath_test_images = r"C:\Users\clvco\Documents\Code\K-space_rl-data\Fashion\fashion_test_norm.npy"
        datapath_test_labels = r"C:\Users\clvco\Documents\Code\K-space_rl-data\Fashion\fashion_test_labels.npy"

        datapath_train_images = r"C:\Users\clvco\Documents\Code\K-space_rl-data\Fashion\fashion_train_norm.npy"
        datapath_train_labels = r"C:\Users\clvco\Documents\Code\K-space_rl-data\Fashion\fashion_train_labels.npy"

        channels = 25
        classes = 10
    else:
        # WINDOWS PATHS
        test_images_filename = "test_images_v2_fixed_norm.npy"
        test_labels_filename = "test_labels.npy"
        train_images_filename = "train_data_norm_v2.npy"
        train_labels_filename = "train_labels.npy"
        channels = 25
        train_fact = 1.0
        test_fact = 1.0
        classes = 10
        # WINDOWS
        base_dir = "C:/Users/clvco/Documents/Code/K-space_rl-data/"
        if not os.path.exists(base_dir):
            # COLAB PATH
            base_dir = '/content'
        if not os.path.exists(base_dir):
            # UBUNTU
            base_dir = '/home/col/data/k-space-rl'
        datapath_test_images = os.path.join(base_dir, test_images_filename)
        datapath_test_labels = os.path.join(base_dir, test_labels_filename)
        datapath_train_images = os.path.join(base_dir, train_images_filename)
        datapath_train_labels = os.path.join(base_dir, train_labels_filename)

    if task == 'MNIST':
        data_x = torch.from_numpy(np.load(datapath_train_images).swapaxes(0,1)).reshape((-1, 28, 28, channels)).float() / train_fact
    else:
        data_x = torch.from_numpy(np.load(datapath_train_images)).reshape((-1, 28, 28, channels)).float() / train_fact
    data_y = torch.from_numpy(np.load(datapath_train_labels)).long()
    if len(data_y.shape) > 1:
        data_y = torch.argmax(data_y, axis=-1)
    num_train = data_x.shape[0]
    indices = list(range(num_train))
    dataset = CustomDataset(data_x, data_y)
    if datapath_test_images is None:
        np.random.seed(3)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[:800], indices[800:]
        test_dataset = CustomDataset(data_x, data_y)
    else:
        data_x_test = torch.from_numpy(np.load(datapath_test_images)).reshape(
            (-1, 28, 28, channels)).float() / test_fact
        data_y_test = torch.from_numpy(np.load(datapath_test_labels))
        if len(data_y_test.shape) > 1:
            data_y_test = torch.argmax(data_y_test, axis=-1)
        num_test = data_x_test.shape[0]
        train_idx, valid_idx = indices, list(range(num_test))
        test_dataset = CustomDataset(data_x_test, data_y_test)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, channels, classes


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
