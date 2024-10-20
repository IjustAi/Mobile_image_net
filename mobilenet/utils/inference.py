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
from tqdm import tqdm

def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model =  mobilenetv2().to(device)

    _, _, testset = get_data_loaders()
    test_loader_len = len(testset)
    
    criterion = nn.CrossEntropyLoss()

    start_epoch, optimizer_state = load_checkpoint(model, optimizer=None, filename='mobilenetv2_checkpoint_50.pth', directory='/Users/chenyufeng/desktop/mobilenet/checkpoint')
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

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(all_targets), yticklabels=np.unique(all_targets))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    torch.mps.empty_cache()  

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, test_loader_len + 1), losses, marker='o', linestyle='-', color='b', label='Test Loss per Batch')
    plt.title('Test Loss Over Batches')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.xticks(range(1, test_loader_len + 1))
    plt.savefig('test_loss.png') 
    plt.show()

if __name__ =='__main__':
    main()

