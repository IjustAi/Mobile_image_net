import os 
import sys
import math 
import torch 
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split
from dataloaders import get_data_loaders, save_checkpoint, load_checkpoint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.mobilenetV2 import mobilenetv2
from math import cos, pi
from tqdm import tqdm 

def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = mobilenetv2().to(device)


    train_loader, val_loader, test_loader = get_data_loaders()
    test_loader_len = len(test_loader)  

    criterion = nn.CrossEntropyLoss()

    start_epoch, optimizer_state = load_checkpoint(model, optimizer=None, filename='mobilenetv2_checkpoint_12.pth', directory='/Users/chenyufeng/desktop/mobilenet/checkpoint')
    if optimizer_state:
        print("Checkpoint loaded.")

    model.eval() 

    sample_index = 24
    for inputs, targets in test_loader:  
        sample_input = inputs[sample_index].unsqueeze(0).to(device)  
        sample_target = targets[sample_index].item()
        break 

    with torch.no_grad():
        sample_output = model(sample_input)
        _, sample_predicted = sample_output.max(1)

    plt.imshow(sample_input.cpu().squeeze().permute(1, 2, 0))  
    plt.title(f'True Label: {sample_target}, Predicted: {sample_predicted.item()}')
    plt.axis('off')
    plt.show()

if __name__ =='__main__':
    main()



