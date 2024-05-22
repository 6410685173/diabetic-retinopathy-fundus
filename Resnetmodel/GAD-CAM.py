import torchvision.transforms as transforms
import torch
from DataGenerator import DatasetGenerator,DatasetGenerator2
from model import ResNet50_DR,VGG19_DR,ResNet50_DR_V2
from modelauto import AutoRes50
from torch.nn.parallel import DataParallel

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from FGRNET import *


test_dir = ["D:\CN341\Basemodel\Resnetmodel\messidor2preprocess\messidor-2\messidor-2\preprocess"]
label_test_file ="D:\CN341\Basemodel\Resnetmodel\idrid-dataset/test_messi.csv"

# ================================================================ # 
# Data augmentation and normalization for training
# Just normalization for validation

test_transform = transforms.Compose([
      transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
""" resolution = 448
test_transform = transforms.Compose([
          transforms.Resize(resolution),
          
          transforms.ToTensor(),
          
      ]) """
# ================================================================ # 
model_name = 'ResNet50_DR_V2' 
Imagedataset_test = DatasetGenerator(data_dir=test_dir, list_file=label_test_file,
                           n_class= 5,transform=test_transform)

dataloaders_test= torch.utils.data.DataLoader(Imagedataset_test, batch_size=1, shuffle=False, num_workers=0)
dataset_train_sizes = len(dataloaders_test)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print("Device: ", device)
print(f"test size: {len(Imagedataset_test)}")
# ================================================================ # 
model = ResNet50_DR_V2(n_classes=6,name="resnetauto_test")
model = DataParallel(model)
model.load_state_dict(torch.load('D:\CN341\Basemodel\Resnetmodel\checkpoint-Resnet-6class\ResNet50_DR_V2/best_model_state_31.ckpt',map_location=torch.device('cpu')))
model.to(device)
model.eval()

#model = UNet16()
#model.load_state_dict(torch.load('D:\CN341\Basemodel\Resnetmodel\FGR_Checkpoints/best_model_state (1).ckpt', map_location=device))
#model.to(device)
#model.eval()

j = 0
#for part1,part2,part3,part4,label in dataloaders_test:
for img,label in dataloaders_test:
    if j == 52:
        break
    
    print(label)
    img = img.to(device)
    #part1 = part1.to(device)
    #part2 = part2.to(device)
    #part3 = part3.to(device)
    #part4 = part4.to(device) 
    #outputs,decode1,decode2,decode3,decode4 = model(part1,part2,part3,part4)
    outputs = model(img)
    preds = outputs.argmax(dim=1)
    #print(outputs)
    print(preds)
    #print(outputs[:,preds])
    outputs[:,preds].backward()
    gradients = model.module.get_activations_gradient()
    #print(gradients.size())

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    #activations = model.module.get_activations(part1,part2,part3,part4).detach()
    activations = model.module.get_activations(img).detach()
    #print(activations.size())
    # weight the channels by corresponding gradients
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()
    # draw the heatmap
    plt.matshow(heatmap.squeeze())
    plt.savefig("heat")

    import cv2
    img = cv2.imread(Imagedataset_test.get_path(j))
    
    #print("Heatmap type:", type(heatmap))
    #print("Heatmap shape:", heatmap.shape)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(f'D:\CN341\Basemodel\GRAD_CAM/resnet/GC_{j}.jpg', superimposed_img)
    j+=1
"""
path = "D:\CN341\Basemodel\Resnetmodel\messidor2preprocess\messidor-2\messidor-2\preprocess/20060529_57430_0100_PP.png"
img= Image.open(path)
img = torch.unsqueeze(test_transform(img),0)

img = img.to(device)
outputs = model(img)
preds = outputs.argmax(dim=1)
print(outputs)
print(preds)
print(outputs[:,preds])
outputs[:,preds].backward()
gradients = model.get_activations_gradient()
# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
# get the activations of the last convolutional layer
activations = model.get_activations(img).detach()
# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze().cpu()
# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)
# normalize the heatmap
heatmap /= torch.max(heatmap)
heatmap = heatmap.numpy()
# draw the heatmap
plt.matshow(heatmap.squeeze())
plt.savefig("heat")
import cv2
img = cv2.imread(path)

print("Heatmap type:", type(heatmap))
print("Heatmap shape:", heatmap.shape)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite(f'./map-test.jpg', superimposed_img)
"""