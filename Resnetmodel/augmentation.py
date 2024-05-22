from torchvision import datasets, transforms
from torchvision.utils import save_image
import random
import os
import itertools
from DataGenerator import DatasetGenerator
import pandas as pd


random.seed(42)
images_dir ='D:\CN341\Basemodel\Resnetmodel\idrid-dataset\Imagenes\Imagenes' #  5 class
label_file ='D:\CN341\Basemodel\Resnetmodel\idrid-dataset\data.csv'

transform = transforms.Compose([
    transforms.ToTensor(),           
])

augmentation_transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    
])

image_datasets = DatasetGenerator(data_dir=images_dir, list_file=label_file,
                             n_class= 5,transform=transform)

print(image_datasets.classes)
data = pd.read_csv(label_file)[["image","class"]]

class_counts = [0] * len(image_datasets.classes)
for c in range(len(image_datasets.classes)):
    class_counts[c] += len(data.loc[data["class"]== c].index)
print(class_counts)
majority_class_count = max(class_counts)
minority_classes = [idx for idx, count in enumerate(class_counts) if count < majority_class_count]
augmented_minority_indices = {}

for minority_class in minority_classes:
    minority_indices = [idx for idx in data.loc[data["class"] == minority_class].index]
    augmented_minority_indices[minority_class] = random.choices(minority_indices, k=majority_class_count - len(minority_indices))  # Sample with replacement


output_dir = "D:\CN341\CNN_Basemodel\Resnetmodel/augment"
df = pd.DataFrame(columns=["image","class"])
i = 0 
for c in augmented_minority_indices:
    index_counters = {}
    
    for idx in augmented_minority_indices[c]:
        image, label = image_datasets[idx]
        augmented_image = augmentation_transform(image)
       
        if idx not in index_counters:
            index_counters[idx] = itertools.count()

        filename = f'aug_image_{label}_{idx}_{next(index_counters[idx])}.jpeg'
        df.loc[i,'image'] = filename[:-5]
        df.loc[i,'class'] = label
        save_image(augmented_image, os.path.join(output_dir, filename ))
        i=i+1

df.to_csv("augmented.csv")
            
