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
import matplotlib.pyplot as plt
from EfficientNet import EfficientNet_b3
from test_datagenerator import DatasetGenerator
# ================================================================ # 
def test_model(model, dataloaders, device):
  
  print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
  model = model.eval()
  i=1
  df = pd.DataFrame(columns=["No","predict","normal","mild","moderate","severe","PDR","Others"])
  with torch.no_grad():
    for image,label in tqdm(dataloaders):
      inputs = image.to(device)
      #labels = labels.to(device)
      outputs = model(inputs)
      _,preds = torch.max(outputs, dim=1)
      
      df.loc[i] =  [i,preds.data[0].cpu().numpy(),outputs[0][0].item(),outputs[0][1].item(),outputs[0][2].item(),outputs[0][3].item(),outputs[0][4].item(),outputs[0][5].item()]
      i +=1
  df.to_csv("result.csv")

# ================================================================ # 

if __name__ == '__main__':
  
  test_dir = []
  label_test_file =""

  # ================================================================ # 
  # Data augmentation and normalization for training
  # Just normalization for validation
  
  test_transform = transforms.Compose([
        transforms.Resize(320, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  # ================================================================ # 

  model_name = 'EfficientNet_b3' 

  Imagedataset_test = DatasetGenerator(data_dir=test_dir, list_file=label_test_file,
                             n_class= 6,transform=test_transform)
  
  dataloaders_test= torch.utils.data.DataLoader(Imagedataset_test, batch_size=1, shuffle=False, num_workers=2)
  dataset_train_sizes = len(dataloaders_test)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(torch.cuda.is_available())
  print("Device: ", device)
  print(f"test size: {len(Imagedataset_test)}")

  # ================================================================ # 

  model_name = 'EfficientNet_b3' 
  # ================================================================ # 
  model = EfficientNet_b3(n_classes=6,name=model_name)
  model.load_state_dict(torch.load('',map_location=torch.device('cpu')))
  model.to(device)
  test_model(model, dataloaders_test, device)
