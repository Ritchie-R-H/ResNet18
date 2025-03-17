import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import pickle
import os
from PIL import Image
from utils import cutmix_data
import utils


from matplotlib import pyplot as plt

def oversample_class(data, labels, target_class, factor=2):
    """ Oversample a specific class in the dataset by a given factor."""
    class_indices = np.where(labels == target_class)[0]
    extra_data = data[class_indices]
    extra_labels = labels[class_indices]

    for _ in range(factor - 1):
        data = np.concatenate((data, extra_data), axis=0)
        labels = np.concatenate((labels, extra_labels), axis=0)
    
    return data, labels

def load_data(batch_size = 64):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    # download CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # dataloader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

def show_image(img_tensor, label):
    # labels
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    img = img_tensor.permute(1, 2, 0).numpy()
    
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    img = (img * std + mean) * 255.0
    img = img.astype(np.uint8)
    
    # show img
    plt.imshow(img)
    plt.title(f"Label: {classes[label]}")
    plt.axis('off')  # Hide axes for better image display
    plt.show()

if __name__ == "__main__":

    trainloader, testloader = load_data()
    print(f"Number of training batches: {len(trainloader)}")
    print(f"Number of testing batches: {len(testloader)}")
    print(f"Number of training samples: {len(trainloader.dataset)}")
    print(f"Number of testing samples: {len(testloader.dataset)}")
    print(np.max(trainloader.dataset[0][0].numpy()))
    print(np.min(trainloader.dataset[0][0].numpy()))

    # show 10 images from the training set
    for i in range(10):
        image, label = trainloader.dataset[i]
        show_image(image, label)

    for i in range(128, 138):
        image, label = trainloader.dataset[i]
        show_image(image, label)