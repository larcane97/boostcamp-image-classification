import torch
import torchvision
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
import os
import albumentations
import albumentations.pytorch
import ttach as tta

class MaskAugmentation:
    def __init__(self,image_size=[256,256]):
        self.transforms={
            'train': albumentations.Compose([
                            albumentations.Resize(image_size[0],image_size[1]),
                            #albumentations.RandomCrop(224,224),
                            albumentations.CenterCrop(int(image_size[0]/0.875),int(image_size[1]/0.875)),
                            albumentations.ColorJitter(),
                            albumentations.HorizontalFlip(),
                            albumentations.Normalize(),
                            albumentations.pytorch.ToTensorV2(),
                           ]),
            'validation' : albumentations.Compose([
                            albumentations.Resize(image_size[0],image_size[1]),
                            albumentations.Normalize(),
                            albumentations.pytorch.ToTensorV2()]),
            'tta' : tta.Compose([
                            tta.HorizontalFlip()
            ])
        }
    
    def __call__(self,mode='train'):
        if not mode in self.transforms.keys():
            raise ValueError(f'mode have to be {list(self.transforms.keys())}')
        return self.transforms[mode]

class MaskDataSet(torch.utils.data.Dataset):
  
    def __init__(self,train_csv='/opt/ml/input/data/allconcat.csv',multi_label=False,transform=None):
        self.train_csv=pathlib.Path(train_csv)
        self.transform=transform
        self.multi_label=multi_label

        self.image_paths, self.label_classes, self.labels = self.read_csv_file(self.train_csv)
        
    def __len__(self):
        return len(self.file_paths)    
    
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # image data 불러오기
        image = np.array(Image.open(self.image_paths[idx]))
        if self.transform:
            print('transformed..')
            image = self.transform(image=image)['image']
        
        if self.multi_label:
            y = np.array(self.labels[idx])
        else:
            y = np.array(self.label_classes[idx])
        
        return image,y
        
    def set_transform(self,transform):
        self.transform=transform
    
    def read_csv_file(self,train_dir):
        ''' Return file path using directory path in csv_file  '''
        data_pd = pd.read_csv(train_dir,encoding='utf-8')

        return data_pd['path'], data_pd['label'], list(zip(data_pd['gender'],data_pd['age'],data_pd['mask']))

class TestDataset(torch.utils.data.Dataset):
    def __init__(self,image_paths,transform=albumentations.Resize(256,256),pseudo_label=None,pseudo_label_path=None):
        self.image_paths = image_paths
        self.transform = transform
        self.pseudo_label_path=pseudo_label_path
        self.pseudo_label = pseudo_label
        if pseudo_label != None:
            self.labels = pseudo_label
        elif pseudo_label_path:
            self.labels = pd.read_csv(pseudo_label_path)['ans']
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image = np.array(Image.open(self.image_paths[idx]))
        
        if self.transform:
            image = self.transform(image=image)['image'].float()
        
        if self.pseudo_label_path or self.pseudo_label != None:
            return image, self.labels[idx]
        return image

class CustomSubset(torch.utils.data.Subset):

    def __init__(self, dataset, indices,transform=None):
        super(CustomSubset,self).__init__(dataset,indices)
        self.dataset = dataset
        self.indices = indices
        self.transform=transform

    def __getitem__(self, idx):
        x,y = self.dataset[self.indices[idx]]
        x = self.transform(image=np.array(x))['image'].float()
        return x,y

    def __len__(self):
        return len(self.indices)