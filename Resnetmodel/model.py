import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights, Inception3, Inception_V3_Weights
import torch.nn as nn
import torch



""" class ResNet50_DR(nn.Module):
    def __init__(self, n_classes,name):
        super(ResNet50_DR,self).__init__()
        self.n_classes = n_classes
        self.name = name
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # use pretained weight
        num_features = self.resnet.fc.out_features
        self.classifier = nn.Sequential(
            nn.Linear(num_features,512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, input):
        x = self.resnet(input)
        output = self.classifier(x)
        return output """
    

class VGG19_DR(nn.Module):
    def __init__(self, n_classes,name):
        super(VGG19_DR,self).__init__()
        self.n_classes = n_classes
        self.name = name
        self.vgg19 = models.vgg19(pretrained=False) 
        num_features = self.vgg19.classifier._modules['6'].out_features 
        self.classifier = nn.Sequential(
            nn.Linear(num_features,512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, input):
        x = self.vgg19(input)
        output = self.classifier(x)
        return output
    
class IncecptionV3(nn.Module):
    def __init__(self, n_classes,name):
        super(IncecptionV3,self).__init__()
        self.inception = models.inception_v3()
        self.name = name
        # Modify the final fully connected layer of the InceptionV3 model
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, n_classes)
        
        # Add a custom fully connected layer for the primary net
        #self.inception.fc = nn.Linear(num_ftrs, 2)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Forward pass through the inception model
        outputs = self.inception(x)
        return outputs

class ResNet50_DR(nn.Module):
    def __init__(self, n_classes,name):
        super(ResNet50_DR,self).__init__()
        self.n_classes = n_classes
        self.name = name
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # use pretained weight
        num_features = self.resnet.fc.out_features

        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = self.resnet.avgpool
        self.fc = self.resnet.fc

        self.classifier = nn.Sequential(
            nn.Linear(num_features,n_classes)
        )
        # placeholder for the gradients
        self.gradients = None
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # register the hook
        #h = x.register_hook(self.activations_hook)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.classifier(x)
        return output
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        a = self.layer4(x)
        return a
    
