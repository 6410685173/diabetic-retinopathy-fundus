import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import torch.autograd
import pathlib 
import torch, torchvision
from matplotlib import rc
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import save_image
from DataGenerator import DatasetGenerator
from model import ResNet50_DR,VGG19_DR,ResNet50_DR_V2,IncecptionV3
import matplotlib.pyplot as plt
from modelrexnet import ReXNetV2
# ================================================================ # 
def train_epoch(model,dataloaders,loss_fn,optimizer,device,scheduler,n_examples):
  model = model.train()
  losses = []
  correct_predictions = 0
  i = 1
  for image,label in tqdm(dataloaders):
    inputs = image.to(device)
    labels = label.to(device)
    print(labels)
    
    outputs,_ = model(inputs)   
    print(outputs)
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, labels)
    correct_predictions += torch.sum(preds == labels)
    losses.append(loss.item())    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()

  return model, correct_predictions.double() / n_examples ,np.mean(losses) # 

# ================================================================ # 
def eval_model(model, dataloaders, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for inputs, labels in tqdm(dataloaders):
      inputs = inputs.to(device)
      labels = labels.to(device)
      print(labels)
      outputs = model(inputs)
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, labels)
      correct_predictions += torch.sum(preds == labels)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses) 

# ================================================================ # 
def checkpoint_path(filename,model_name):
  
  checkpoint_folderpath = pathlib.Path(f'D:\CN341\Basemodel\Resnetmodel/checkpoint-Inception-6class/{model_name}')
  print(checkpoint_folderpath)
  checkpoint_folderpath.mkdir(exist_ok=True,parents=True)
  return checkpoint_folderpath/filename
# ================================================================ # 

def train_model(model, dataloaders_train, dataloaders_val,  dataset_sizes_train,  dataset_sizes_val, device, n_epochs=50): # train ต่อจาก epoch ที่18
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss(reduction='mean').to(device)
    loss_MSE = nn.MSELoss().to(device)
    best_model_path = checkpoint_path('best_model_state.ckpt',model.name)
   
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

      torch.save(model.state_dict(), checkpoint_path('best_model_state_'+str(epoch)+'.ckpt',model.name))
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
# ================================================================ #  

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


if __name__ == '__main__':
  
  train_dir =['D:\CN341\Basemodel\Resnetmodel\messidor2preprocess\messidor-2\messidor-2\preprocess'] #  6 class
  label_train_file ='D:\CN341\Basemodel\Resnetmodel\idrid-dataset/test_messi.csv'
  val_dir =['D:\CN341\Basemodel\Resnetmodel\idrid-dataset\Imagenes\Imagenes'] #  6 class
  label_val_file ='D:\CN341\Basemodel\Resnetmodel\idrid-dataset/train5.csv'
  

  # ================================================================ # 
  # Data augmentation and normalization for training
  # Just normalization for validation
  
  train_transforms = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  val_trasform = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  # ================================================================ # 

  

  Imagedataset_train = DatasetGenerator(data_dir=train_dir, list_file=label_train_file,
                             n_class= 5,transform=train_transforms)

  Imagedataset_val = DatasetGenerator(data_dir=val_dir, list_file=label_val_file,
                             n_class= 5,transform=val_trasform)
  
  dataloaders_train= torch.utils.data.DataLoader(Imagedataset_train, batch_size=4, shuffle=False, num_workers=2)
  dataloaders_val= torch.utils.data.DataLoader(Imagedataset_val, batch_size=4, shuffle=False, num_workers=2)

  dataset_train_sizes = len(Imagedataset_train)
  dataset_val_sizes = len(Imagedataset_val)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # ================================================================ # 
  print(torch.cuda.is_available())
  print("Device: ", device)
  print(f"train size: {len(Imagedataset_train)}")
  print(f"val size: {len(Imagedataset_val)}")

  model_name = 'IncecptionV3' 
  # ================================================================ # 
  model = IncecptionV3(n_classes=5,name=model_name)
  model.to(device)
  # ================================================================ # 
  model = train_model(model,dataloaders_train, dataloaders_val, dataset_train_sizes, dataset_val_sizes, device, n_epochs=50)
  # ================================================================ # 

