import os 
import math 
import torch 
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split
from utils.dataloaders import get_data_loaders, save_checkpoint, load_checkpoint
from model.mobilenetV2 import mobilenetv2
from model.mobilenetCA import mobilenetCA
from model.mobilenetV4 import mobilenetv4
from math import cos, pi
from tqdm import tqdm  

def adjust_learning_rate(optimizer, epoch, iteration, num_iter, total_epochs):
    lr = optimizer.param_groups[0]['lr']
    warmup_epoch = 3
    warmup_iter = iteration + epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = total_epochs * num_iter 

    if warmup_iter > 0:
        lr = lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    else:
        lr = lr  

    if epoch < warmup_epoch:
        lr = lr * current_iter / warmup_iter if warmup_iter > 0 else lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch, device,total_epochs):
    model.train()
    running_loss = 0.0
    with tqdm(total=train_loader_len, desc=f'Epoch {epoch + 1}/{total_epochs}', unit='batch') as pbar:
        for i, (inputs, targets) in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, i, train_loader_len, total_epochs)
            inputs, targets = inputs.to(device), targets.to(device) 

            optimizer.zero_grad()  

            output = model(inputs)
            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            pbar.set_postfix(loss=loss.item()) 
            pbar.update(1)  

            if i % 100 == 99:  
                print(f'Epoch [{epoch + 1}], Step [{i + 1}/{train_loader_len}], Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / train_loader_len
    print(f'Epoch [{epoch+1}] Training Loss: {avg_loss:.4f}')
    torch.mps.empty_cache()
    return avg_loss


def validate(val_loader, val_loader_len, model, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for i, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device) 
        with torch.no_grad():
            output = model(inputs)
            loss = criterion(output, targets)
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = total_loss / val_loader_len
    accuracy = correct / total
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    torch.mps.empty_cache()
    return avg_loss


def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    #model = mobilenetv2().to(device)
    #model = mobilenetCA().to(device)
    model = mobilenetv4().to(device)


    trainloader, validloader, _= get_data_loaders()

    train_loader_len = len(trainloader)
    val_loader_len = len(validloader)

    criterion = nn.CrossEntropyLoss()
    lr = 0.05
    momentum = 0.9
    weight_decay = 4e-5
    total_epochs=50
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if model.__class__.__name__ == "MobileNetv2":
        checkpoint_path = '/Users/chenyufeng/desktop/mobilenet/checkpoint/mobilenetv2_checkpoint_100.pth'
        directory = '/Users/chenyufeng/desktop/mobilenet/checkpoint'
        filename = 'mobilenetv2_checkpoint_50.pth'
        model_name = 'mobilenetv2'

    elif model.__class__.__name__ == "MobileNetCA":
        checkpoint_path = '/Users/chenyufeng/desktop/mobilenet/checkpoint_CA/mobilenet_CA_checkpoint_100.pth'
        directory = '/Users/chenyufeng/desktop/mobilenet/checkpoint_CA'
        filename = 'mobilenet_CA_checkpoint_50.pth'
        model_name = 'mobilenetCA'

    else:
        checkpoint_path = '/Users/chenyufeng/desktop/mobilenet/checkpoint_V4/mobilenetV4_checkpoint_19.pth'
        directory = '/Users/chenyufeng/desktop/mobilenet/checkpoint_V4'
        filename = 'mobilenetv4_checkpoint_19.pth'
        model_name = 'mobilenetv4'

    if os.path.isfile(checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, filename=filename, directory=directory)
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch + 1}")

    else:
        print(f"Checkpoint loaded. Resuming training from epoch 0")
        start_epoch = 0  

    training_losses = []  
    validation_losses = []

    for epoch in range(start_epoch,total_epochs):  
        print(f"Epoch {epoch + 1}:")
        avg_train_loss = train(trainloader, train_loader_len, model, criterion, optimizer, epoch, device,total_epochs)
        avg_val_loss = validate(validloader, val_loader_len, model, criterion, device)

        training_losses.append(avg_train_loss)  
        validation_losses.append(avg_val_loss)  
        save_checkpoint(model, optimizer, epoch, criterion, filename=f'{model_name}_checkpoint_{epoch + 1}.pth', directory=directory)

    
    epochs_range = range(start_epoch + 1, total_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, training_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.plot(epochs_range, validation_losses, marker='o', linestyle='-', color='r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.xticks(epochs_range) 
    plt.savefig('training_validation_loss.png')  
    plt.show()

if __name__ == "__main__":
    main()