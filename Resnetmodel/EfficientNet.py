import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
import torch.nn as nn




class EfficientNet_b3(nn.Module):
    def __init__(self, n_classes,name):
        super(EfficientNet_b3,self).__init__()
        self.n_classes = n_classes
        self.name = name
        self.efficientnet = efficientnet_b3(EfficientNet_B3_Weights.IMAGENET1K_V1) # use pretained weight
        
        self.classifier = nn.Sequential(
            nn.Linear(1000,n_classes),
        )

    def forward(self, input):
        x = self.efficientnet(input)
        output = self.classifier(x)
        return output
    

