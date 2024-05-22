from torch.utils.data import Dataset
import torch
import pandas as pd
import albumentations
import os
import numpy as np
from PIL import Image
def add_path(data_dir,name):

  path = os.path.join(data_dir, name + '.jpg')
  if os.path.exists(path):
      return path
  path = os.path.join(data_dir, name + '.jpeg')
  if os.path.exists(path):
      return path
  path = os.path.join(data_dir, name + '.JPG')
  if os.path.exists(path):
      return path
  path = os.path.join(data_dir, name + '.png')
  if os.path.exists(path):
      return path


def load_excel(data_dir, list_file,n_class):    
    
  image_paths = []
  labels = []
  df_tmp = pd.read_csv(list_file)
  augmented_indices = {}
  class_counts = [0]*n_class
  for c in range(n_class):
    class_counts[c] += len(df_tmp.loc[df_tmp["class"] == c].index)
    augmented_indices[c] = [idx for idx in df_tmp.loc[df_tmp["class"] == c].index]
  print(class_counts)
  minority_class = min(class_counts)
  print(minority_class)
  for ix in range(minority_class):
    for jx in range(len(class_counts)):
      p = ""
      image_name = df_tmp["image"][augmented_indices[jx][ix]]
      label = df_tmp["class"][augmented_indices[jx][ix]]
      
      for i in range(len(data_dir)):
        
        p = add_path(data_dir[i],image_name)
        if p != None :
          break
        
      if p == None:
        print(f"Image not found for {image_name}")
      else:
        image_paths.append(p)
        labels.append(label)
  return image_paths,labels


class DatasetGenerator(Dataset):

  def __init__(self, data_dir, list_file, transform=None, n_class=6):

    image_names,labels = load_excel(data_dir, list_file,n_class)

    self.image_names = image_names
    self.classes = list(set(labels))
    self.labels = labels
    self.n_class = n_class
    self.transform = transform

  def __getitem__(self, index):

    image_name = self.image_names[index]
    label = self.labels[index]
    image = Image.open(image_name)

    if self.transform is not None:
      image = self.transform(image)
     

    return image,label

  def get_path(self,index):
    return self.image_names[index]


  def __len__(self):
     return len(self.image_names)


class DatasetGenerator2(Dataset):

  def __init__(self, data_dir, list_file, transform=None, n_class=6):

    image_names,labels = load_excel(data_dir, list_file,n_class)

    self.image_names = image_names
    self.classes = list(set(labels))
    self.labels = labels
    self.n_class = n_class
    self.transform = transform
    self.CLAHE = albumentations.Compose([ albumentations.CLAHE(clip_limit=(1,4), p= 1),])


  def __getitem__(self, index):

    image_name = self.image_names[index]
    label = self.labels[index]
    image = Image.open(image_name)


    if self.transform is not None:
        np_image = np.array(image)

        # Apply CLAHE transformation
        transformed_CLAHE = self.CLAHE(image=np_image)['image']
        transformed_CLAHE = Image.fromarray(transformed_CLAHE)
        image = self.transform(transformed_CLAHE)

    h = image.shape[1]
    w = image.shape[2]
    part1= image[:,:(h//2),:(w//2)]
    part2= image[:,:(h//2),(w//2):]
    part3= image[:,(h//2):,:(w//2)]
    part4= image[:,(h//2):,(w//2):]

    return part1,part2,part3,part4,label

  def get_path(self,index):
    return self.image_names[index]


  def __len__(self):
     return len(self.image_names)