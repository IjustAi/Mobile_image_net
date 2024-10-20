import os
import torch
import torchvision
import warnings
from torchvision import transforms
from torch.utils.data import random_split

warnings.filterwarnings("ignore", category=Warning)

def get_data_loaders(batch_size=64, valid_size=0.1):

    seed = 42
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='/Users/chenyufeng/desktop/mobilenet/dataset', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='/Users/chenyufeng/desktop/mobilenet/dataset', train=False, download=True, transform=transform)

    valid_size = int(len(trainset) * valid_size)
    train_size = len(trainset) - valid_size
    train_subset, valid_subset = random_split(trainset, [train_size, valid_size])

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
    validloader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, validloader, testloader

def save_checkpoint(model,optimizer,epoch,loss,filename='checkpoint.pth',directory='/Users/chenyufeng/desktop/mobilenet/checkpoint'):
    os.makedirs(directory,exist_ok=True)

    filepath = os.path.join(directory,filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

def load_checkpoint(model, optimizer, filename, directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None: 
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: {filepath}")
        return epoch, loss
    else:
        print(f"No checkpoint found at '{filepath}'")
        return 0, None  


