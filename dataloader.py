'''
This module contains the method to build dataset and dataload.
'''

from torchvision import transforms, datasets
import torch


data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

def get_data_loader(data_dir, batch_size, num_workers, aug=False):
    """
    Create a DataLoader for the dataset located in data_dir.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of images to load per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        aug (bool): If True, use training data transformations (with augmentation). 
                    If False, use validation data transformations.

    Returns:
        loader (torch.utils.data.DataLoader): DataLoader for the specified dataset.
        dataset (torchvision.datasets.ImageFolder): The dataset object created from the specified directory.
    """
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform["train" if aug else "val"])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size, shuffle=aug,
                                         num_workers=num_workers)

    return loader, dataset
