import os 
import sys
import math 
import torch 
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt  
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from torch.utils.data import random_split
from dataloaders import get_data_loaders, save_checkpoint, load_checkpoint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.mobilenetV2 import mobilenetv2
from model.mobilenetCA import mobilenetCA
from model.mobilenetV4 import mobilenetv4
from tqdm import tqdm

def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model =  mobilenetCA().to(device)

    _, _, testset = get_data_loaders()
    test_loader_len = len(testset)
    
    criterion = nn.CrossEntropyLoss()

    start_epoch, optimizer_state = load_checkpoint(model, optimizer=None, filename='mobilenetCA_checkpoint_50.pth', directory='/Users/chenyufeng/desktop/mobilenet/checkpoint_CA')
    if optimizer_state:
        print(f"Checkpoint loaded.")
    
    model.eval() 
    total_loss = 0 
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    losses = []

    with tqdm(total=test_loader_len, desc=f'Testing', unit='batch') as pbar:
        for i, (inputs, targets) in enumerate(testset): 
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():  
                output = model(inputs)
                loss = criterion(output, targets)
                total_loss += loss.item()
                losses.append(loss.item())

                _, predicted = output.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            pbar.update(1)  

    avg_loss = total_loss / test_loader_len
    accuracy = correct / total
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    report = classification_report(all_targets, all_predictions, digits=4)
    print("\nClassification Report:\n", report)

    cm = confusion_matrix(all_targets, all_predictions)
    

if __name__ =='__main__':
    main()

