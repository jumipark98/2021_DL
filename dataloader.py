import os
import numpy as np
from albumentations import CLAHE,Compose
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torch.utils.data import Dataset
import json 


class CustomDataset():
    def __init__(self,phase):
        self.phase = phase
        self.png_root = '/SSD3/jumi/UAVID/PNGImages/'+phase.lower()
        self.annot_root = '/SSD3/jumi/UAVID/Annotations/'+phase.lower()
        self.fn = os.listdir(self.png_root)
        self.clahe = CLAHE(p=0.8)
        self.img_transform = transforms.Compose([
                                                  transforms.Resize((600,1200)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
        self.tgt_transform = transforms.Compose([
                                                  transforms.Resize((600,1200), interpolation=Image.NEAREST),
                                                  np.asarray,
                                                  torch.LongTensor
                                                 ])
        
    def __len__(self):
        return len(self.fn)

    def __getitem__(self, idx):
    
        img_name = self.fn[idx]
      
        png_img = cv2.imread(os.path.join(self.png_root, img_name))
        data = {"image":np.array(png_img)}
        aug = self.clahe(**data)
        png_img = aug["image"]
        png_img = Image.fromarray(png_img)
        annot_img = Image.open(os.path.join(self.annot_root, img_name))
        
        png = self.img_transform(png_img)
        annot = self.tgt_transform(annot_img)
                                  
        return png,annot,img_name



class CustomDataset2():
    def __init__(self,phase):
        self.phase = phase
        self.png_root = '/SSD3/jumi/UAVID/PNGImages/'+phase.lower()
        self.annot_root = '/SSD3/jumi/UAVID/Annotations2/'+phase.lower()
        self.fn = os.listdir(self.png_root)
        self.clahe = CLAHE(p=0.8)
        self.img_transform = transforms.Compose([
                                                  transforms.
                                                  transforms.Resize((500,1000)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
        self.tgt_transform = transforms.Compose([
                                                  transforms.Resize((500,1000), interpolation=Image.NEAREST),
                                                  np.asarray,
                                                  torch.LongTensor
                                                 ])
        
    def __len__(self):
        return len(self.fn)
    

    def __getitem__(self, idx):
    
        img_name = self.fn[idx]
      
        png_img = cv2.imread(os.path.join(self.png_root, img_name))
        data = {"image":np.array(png_img)}
        aug = self.clahe(**data)
        png_img = aug["image"]
        png_img = Image.fromarray(png_img)
        annot_img = Image.open(os.path.join(self.annot_root, img_name))
        
        png = self.img_transform(png_img)
        annot = self.tgt_transform(annot_img)
                                  
        return png,annot,img_name

class CustomDataset3():
    def __init__(self,phase):
        self.phase = phase
        # /SSD3/jumi/ASCP/PNGImages/train
        self.png_root = '/SSD3/jumi/ASCP/PNGImages/'+phase.lower()
        self.annot_root = '/SSD3/jumi/ASCP/Annotations2/'+phase.lower()
        self.fn = os.listdir(self.png_root)
        self.clahe = CLAHE(p=0.8)
        self.img_transform = transforms.Compose([
                                                  transforms.
                                                  transforms.Resize((500,1000)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
        self.tgt_transform = transforms.Compose([
                                                  transforms.Resize((500,1000), interpolation=Image.NEAREST),
                                                  np.asarray,
                                                  torch.LongTensor
                                                 ])
        
    def __len__(self):
        return len(self.fn)
    

    def __getitem__(self, idx):
    
        img_name = self.fn[idx]
      
        png_img = cv2.imread(os.path.join(self.png_root, img_name))
        data = {"image":np.array(png_img)}
        aug = self.clahe(**data)
        png_img = aug["image"]
        png_img = Image.fromarray(png_img)
        annot_img = Image.open(os.path.join(self.annot_root, img_name))
        
        png = self.img_transform(png_img)
        annot = self.tgt_transform(annot_img)
                                  
        return png,annot,img_name


class InferDataset():
    def __init__(self):
        self.png_root = '/SSD3/jumi/UAV-DT/Train/'
        self.fn = os.listdir(self.png_root)
        self.clahe = CLAHE(p=0.8)
        self.img_transform = transforms.Compose([
                                                  transforms.Resize((600,1200)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
    def __len__(self):
        return len(self.fn)
    

    def __getitem__(self, idx):
        
        img_name = self.fn[idx]
      
        png_img = cv2.imread(os.path.join(self.png_root, img_name))
        data = {"image":np.array(png_img)}
        aug = self.clahe(**data)
        png_img = aug["image"]
        png_img = Image.fromarray(png_img)
        
        png = self.img_transform(png_img)
                       
        return png,img_name,idx