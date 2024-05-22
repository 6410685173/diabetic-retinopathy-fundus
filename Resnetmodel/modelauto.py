import torchvision.models as models
from torchvision.models import Inception3, Inception_V3_Weights, resnet50, ResNet50_Weights
import torch.nn as nn
from torchvision.utils import save_image
import torch

def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x
    
class Interpolate(nn.Module):
    def __init__(
        self,
        size: int = None,
        scale_factor: int = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interp(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x
       
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        is_deconv: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear"),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    

class ResNet50_Base(nn.Module):
    def __init__(self, n_classes,name,is_deconv: bool = False,):
        super(ResNet50_Base,self).__init__()
        self.n_classes = n_classes
        self.name = name
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # use pretained weight
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.encode1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,      
        )
        # encoder
        self.encode2  = resnet.layer1
        self.encode3 = resnet.layer2
        self.encode4 = resnet.layer3
        self.encode5 = resnet.layer4

        # decoder
        self.center = nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear"),
                ConvRelu(2048, 1024),
            )
        self.decode4 = DecoderBlock(
            2048, 1024, 512, is_deconv
        )
        self.decode3 = DecoderBlock(
            1024, 512, 256, is_deconv
        )
        self.decode2 = DecoderBlock(
            512, 256, 64, is_deconv
        )    
        self.decode1 = nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear"),
                ConvRelu(128,64),
                nn.Conv2d(64, 3, kernel_size=1)
            )
            
        self.avgpool = resnet.avgpool
        
        # placeholder for the gradients
        self.gradients = None
        
    def forward(self, input):
        
        e1 = self.encode1(input)
        e2 = self.encode2(self.maxpool(e1))
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        e5 = self.encode5(e4)
        output = e5
        
        # register the hook
        h = e5.register_hook(self.activations_hook)
        
        c = self.center(e5)
    
        d4 = self.decode4(torch.concat([c,e4],1))
        d3 = self.decode3(torch.concat([d4,e3],1))
        d2 = self.decode2(torch.concat([d3,e2],1))
        recon = self.decode1(torch.concat([d2,e1],1))
        
        return output,recon
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    
    

class AutoRes50(nn.Module):
    def __init__(self, n_classes,name):
        super(AutoRes50,self).__init__()
        self.n_classes = n_classes
        self.name = name
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        #part1
        self.base1 = ResNet50_Base(n_classes,"base1")
        #part2
        self.base2 = ResNet50_Base(n_classes,"base2")
        #part3
        self.base3 = ResNet50_Base(n_classes,"base3")
        #part4
        self.base4 = ResNet50_Base(n_classes,"base4")

        self.avg = resnet.avgpool
        self.fc = resnet.fc

        self.classifier = nn.Sequential(
            nn.Linear(1000,n_classes)
        )

    def forward(self, part1,part2,part3,part4):
        encode1,decode1 = self.base1(part1)
        encode2,decode2 = self.base2(part2)
        encode3,decode3 = self.base3(part3)
        encode4,decode4 = self.base4(part4)

        top = torch.cat([encode1, encode2], dim=3)
        bottom = torch.cat([encode3, encode4], dim=3)
        combine = torch.cat([top, bottom], dim=2)
        x = self.avg(combine)
        x = x.view(x.size(0), -1)
        x = self.fc(x)     
        result = self.classifier(x)


        return result, decode1, decode2, decode3, decode4
    
    def get_activations_gradient(self):
        g1 = self.base1.get_activations_gradient()
        g2 = self.base1.get_activations_gradient()
        g3 = self.base1.get_activations_gradient()
        g4 = self.base1.get_activations_gradient()

        top = torch.cat([g1, g2], dim=3)
        bottom = torch.cat([g3, g4], dim=3)
        combine_gradient = torch.cat([top, bottom], dim=2)
        return combine_gradient
    
    # method for the activation exctraction
    def get_activations(self, part1,part2,part3,part4):
        encode1,decode1 = self.base1(part1)
        encode2,decode2 = self.base2(part2)
        encode3,decode3 = self.base3(part3)
        encode4,decode4 = self.base4(part4)

        top = torch.cat([encode1, encode2], dim=3)
        bottom = torch.cat([encode3, encode4], dim=3)
        combine = torch.cat([top, bottom], dim=2)
        return combine


        
        