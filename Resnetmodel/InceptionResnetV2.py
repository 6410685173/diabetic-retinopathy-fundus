import torchvision.models as models
from torchvision.models import Inception3, Inception_V3_Weights
import torch.nn as nn
import torch
import timm


    
class inception_resnet_v2(nn.Module):
    def __init__(self, n_classes,name,pretrained=True):
        super(inception_resnet_v2,self).__init__()
        self.n_classes = n_classes
        self.name = name
        
        inception_resnet = timm.create_model('inception_resnet_v2.tf_in1k', pretrained=pretrained)
        self.conv_inres = torch.nn.Sequential(*list(inception_resnet.children())[:-3])
        self.global_pool =  inception_resnet.global_pool
        self.head_drop =  inception_resnet.head_drop
        self.classif =  inception_resnet.classif
        
        num_features = inception_resnet.classif.out_features
        self.classifier = nn.Sequential(
            nn.Linear(num_features,n_classes)
        )
        
        self.gradients = None

    def forward(self, input):

        x =  self.conv_inres(input)
        
        #h = x.register_hook(self.activations_hook)
        
        x = self.global_pool(x)
        x = self.head_drop(x)
        x = self.classif(x)
        output = self.classifier(x)

        return output
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        
        self.gradients = grad
        

    # method for the gradient extraction
    def get_activations_gradient(self):
        
        return self.gradients

    