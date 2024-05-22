import timm
import csv
import os
#import cv2
import shutil
import pathlib
import torchvision
import numpy as np
import pandas as pd
from torch import nn, optim, cuda
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import torch.autograd
from pathlib import Path
import torch, torchvision
from matplotlib import rc
#from pylab import rcParams
from PIL.Image import Image
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from matplotlib.ticker import MaxNLocator
from torchvision.datasets import ImageFolder
from torchvision import datasets,models,transforms
from sklearn.model_selection import train_test_split
from Datagenerator import DatasetGenerator

# =================================================================================================================
def checkpoint_path(filename,model_name):

  checkpoint_folderpath = pathlib.Path(f'checkpoint/combinedata2/{model_name}')
  print(checkpoint_folderpath)
  checkpoint_folderpath.mkdir(exist_ok=True,parents=True)
  return checkpoint_folderpath/filename

def save_metrics_to_csv(history, target_folder):
    csv_file = os.path.join(target_folder, "metrics.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"])
        for epoch in range(len(history['train_loss'])):
            writer.writerow([epoch+1, history['train_loss'][epoch], history['train_acc'][epoch],
                             history['val_loss'][epoch], history['val_acc'][epoch]])
            
def train_epoch(model,dataloaders,loss_fn,optimizer,device,scheduler,n_examples):
  model = model.train()
  losses = []
  correct_predictions = 0
  i = 1
  for inputs, labels in tqdm(dataloaders):
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    outputs = model(inputs)
    
    preds = outputs.argmax(dim=-1)
    loss = loss_fn(outputs, labels)
   
    i +=1

    correct_predictions += torch.sum(preds == labels)
  
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()
  return model, correct_predictions.double() / n_examples ,np.mean(losses) #

def eval_model(model, dataloaders, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for inputs, labels in tqdm(dataloaders):
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      preds = outputs.argmax(dim=-1)
      loss = loss_fn(outputs, labels)
      correct_predictions += torch.sum(preds == labels)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)

def train_model(model, dataloaders_train, dataloaders_val,  dataset_sizes_train,  dataset_sizes_val, device, n_epochs=50): # train ต่อจาก epoch ที่18
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss(reduction='mean').to(device)
    best_model_path = checkpoint_path('best_model_state.ckpt',"vision_transformer")
   
    #print(model)
    train_accuracy = []
    train_losses = []
    val_accuracy = []
    val_losses = []
  
    best_accuracy = 0
    for epoch in range(1,n_epochs+1):
      print(f'Epoch {epoch }/{n_epochs}')
      print('-' * 10)
      model, train_acc, train_loss = train_epoch(model,dataloaders_train,loss_fn,optimizer,device,scheduler,n_examples=dataset_sizes_train)
      print(f'Train loss {train_loss} accuracy {train_acc}')
      val_acc, val_loss = eval_model(model,dataloaders_val,loss_fn,device,n_examples=dataset_sizes_val)
      print(f'validation   loss {val_loss} accuracy {val_acc}')
      train_accuracy.append(train_acc.item())
      train_losses.append(train_loss)
      val_accuracy.append(val_acc.item())
      val_losses.append(val_loss)

      torch.save(model.state_dict(), checkpoint_path('best_model_state_'+str(epoch)+'.ckpt',"vision_transformer"))
      if val_acc> best_accuracy:
        torch.save(model.state_dict(), best_model_path)
        best_accuracy = val_acc
    #print(f'Best val accuracy: {best_accuracy}')
    model.load_state_dict(torch.load(best_model_path))
    #print(f"train_accuracy_each_epoch {train_accuracy}")
    #print(f"train_losses_each_epoch {train_losses}")
    #print(f"val_accuracy_each_epoch {val_accuracy}")
    #print(f"val_losses_each_epoch {val_losses}")
    plot_metrics(train_accuracy, val_accuracy, 'Accuracy')
    plot_metrics(train_losses, val_losses, 'Loss')
    
    return model

def plot_metrics(train_metrics, val_metrics, metric_name):
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo-', label=f'Training {metric_name}')
    plt.plot(epochs, val_metrics, 'ro-', label=f'Validation {metric_name}')
    plt.xticks(epochs)
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(metric_name)
    plt.show()



# =================================================================================================================
train_dir =['D:\CN341\Basemodel\Resnetmodel\messidor2preprocess\messidor-2\messidor-2\preprocess'] #  6 class
label_train_file ='D:\CN341\Basemodel\Resnetmodel\idrid-dataset/test_messi.csv'
val_dir =['D:\CN341\Basemodel\Resnetmodel\idrid-dataset\Imagenes\Imagenes'] #  6 class
label_val_file ='D:\CN341\Basemodel\Resnetmodel\idrid-dataset/train5.csv'

# save_file_name = '/content/drive/MyDrive/DR/model'
# checkpoint_path = '/content/drive/MyDrive/DR/model'
resolution = 384
data_transforms = {
      'train_FGR': transforms.Compose([
        transforms.Resize(resolution),
          transforms.CenterCrop(resolution),
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(180),
          transforms.ColorJitter(brightness=0.01,contrast=0.01,hue=0.01,saturation=0.01),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val_FGR': transforms.Compose([
          transforms.Resize(resolution),
          transforms.CenterCrop(resolution),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'test': transforms.Compose([
          transforms.Resize(resolution),
          transforms.CenterCrop(resolution),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

Imagedataset_train = DatasetGenerator(data_dir=train_dir, list_file=label_train_file,
                             n_class= 6,transform=data_transforms["train_FGR"])

Imagedataset_val = DatasetGenerator(data_dir=val_dir, list_file=label_val_file,
                             n_class= 6,transform=data_transforms["val_FGR"])
  
dataloaders_train= torch.utils.data.DataLoader(Imagedataset_train, batch_size=2, shuffle=False, num_workers=2)
dataloaders_val= torch.utils.data.DataLoader(Imagedataset_val, batch_size=2, shuffle=False, num_workers=2)

dataset_train_sizes = len(Imagedataset_train)
dataset_val_sizes = len(Imagedataset_val)

if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if device.type == 'cuda':
    torch.cuda.set_device(0)  # Explicitly set CUDA device 0 before transferring the model
  #timm.list_models('*vit_base*',pretrained=False)

  model = timm.create_model('vit_base_r50_s16_384.orig_in21k_ft_in1k', pretrained=True, num_classes = 6)
  model.to(device)
# ================================================================ #
    
# # ================================================================ #
# base_model, encoder = create_model(model_name,num_classes=len(class_names),device=device)
# # ================================================================ #
  model = train_model(model,dataloaders_train, dataloaders_val, dataset_train_sizes, dataset_val_sizes, device, n_epochs=20)
# ================================================================ #

# ================================================================ #