import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys,os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('ml'))))
from fixresnet.FixRes.imnet_evaluate import resnext_wsl
import timm


class EfficientLite0(nn.Module):
    def __init__(self,name='efficientLite0',device='cpu',num_classes=18):
        super(EfficientLite0,self).__init__()
        self.name=name
        self.device=device
        self.num_classes=num_classes
        self.backbone = timm.create_model('efficientnet_lite0',True).to(device)
        self.sequence = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000,500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500,200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(200,num_classes)).to(device)
    
    def forward(self,x):
        x = self.backbone(x)
        return self.sequence(x)

class MaskResNet(nn.Module):
    def __init__(self,num_classes,device,name='MaskResNet',weight_path='../fixresnet/ResNext101_32x48d_v2.pth'):
        super(MaskResNet,self).__init__()
        self.name=name
        self.num_classes=num_classes
        self.device=device
        self.weight_path = weight_path
        self.backbone = self.load_FixResNeXt_101_32x48d().to(device)
        self.classifier = nn.Linear(1000,num_classes).to(device)

            
    def forward(self,x):
        x = self.backbone(x)
        x = F.relu(x)
        return self.classifier(x)
    
    def load_FixResNeXt_101_32x48d(self):
        model=resnext_wsl.resnext101_32x48d_wsl(progress=True)
        pretrained_dict=torch.load(self.weight_path,map_location=self.device)['model']
        model_dict = model.state_dict()
        for k in model_dict.keys():
            if(('module.'+k) in pretrained_dict.keys()):
                model_dict[k]=pretrained_dict.get(('module.'+k))
        model.load_state_dict(model_dict)
        return model

class MaskViT(nn.Module):
    def __init__(self,num_classes,device,name='MaskViT'):
        super(MaskViT,self).__init__()
        self.num_classes=num_classes
        self.name=name
        self.backbone = timm.create_model('vit_base_patch16_384', pretrained=True).to(device)
        self.sequence = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000,500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500,200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(200,num_classes)).to(device)
  
    def forward(self,x):
        x = self.backbone.forward(x)
        x = self.sequence(x)
        return x

class MaskDenseNetThreeHead(nn.Module):
    def __init__(self,device):
        super(MaskDenseNetThreeHead,self).__init__()
        self.device=device
        self.class_num={
            'mask':3,
            'gender':2,
            'age':3
        }

        
        self.densenet_mask = torchvision.models.densenet161(pretrained=True).to(self.device)
        self.densenet_mask.classifier.register_forward_hook(self.mask_hook)

        self.densenet_gender = torchvision.models.densenet161(pretrained=True).to(self.device)
        self.densenet_gender.classifier.register_forward_hook(self.gender_hook)

        self.densenet_age = torchvision.models.densenet161(pretrained=True).to(self.device)
        self.densenet_age.classifier.register_forward_hook(self.age_hook)
        
        self.classifier = nn.Sequential(
                                         nn.Linear(8,50),
                                         nn.BatchNorm1d(50),
                                         nn.ReLU(),
                                         nn.Linear(50,100),
                                         nn.BatchNorm1d(100),
                                         nn.ReLU(),
                                         nn.Linear(100,18)).to(self.device)
        
    def forward(self,x):
        mask_class = self.densenet_mask.forward(x) # B x 3
        gender_class = self.densenet_gender.forward(x) # B x 2
        age_class = self.densenet_age.forward(x) # B x 3
        concat = torch.cat([mask_class,gender_class,age_class],dim=-1)
        return self.classifier.forward(concat)

    def mask_hook(self,model,input,output):
        output = nn.BatchNorm1d(1000).to(self.device)(output)
        output = nn.Linear(1000,3).to(self.device)(output)
        return output
    def gender_hook(self,model,input,output):
        output = nn.BatchNorm1d(1000).to(self.device)(output)
        output = nn.Linear(1000,2).to(self.device)(output)
        return output
    def age_hook(self,model,input,output):
        output = nn.BatchNorm1d(1000).to(self.device)(output)
        output = nn.Linear(1000,3).to(self.device)(output)
        return output

class MaskDenseNet(nn.Module):
    def __init__(self,device='cpu'):
        super(MaskDenseNet,self).__init__()
        self.device=device
        self.densenet = torchvision.models.densenet161(pretrained=True).to(self.device)
        self.linear = nn.Linear(1000,18).to(self.device)
    
    def forward(self,x):
        x = self.densenet(x)
        x = nn.BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(self.device)(x)
        return self.linear(x)
