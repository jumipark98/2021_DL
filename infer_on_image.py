import os 
from dataloader import InferDataset
import torch 
import numpy as np
from utils.draw_segmap import * 
from torchvision.utils import save_image 
from modeling.deeplab import * 
import cv2
from tqdm import tqdm
import os
import numpy as np
from albumentations import CLAHE,Compose
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torch.utils.data import Dataset
import json 

class Inferer():
    def __init__(self):
        self.device = torch.device('cuda:2')
        self.model = DeepLab(num_classes=3,
                    backbone='mobilenet',
                    output_stride=8)
        ckpt = torch.load('run/class3/model_best.pth')['state_dict']
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.clahe = CLAHE(p=0.8)
        self.img_transform = transforms.Compose([
                                                  transforms.Resize((600,1200)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])

    def predict(self,frame,k):
        h,w,c = frame.shape
        data = {"image":(frame)}
        aug = self.clahe(**data)
        png_img = aug["image"]
        png_img = Image.fromarray(png_img)
        png = self.img_transform(png_img)
        png = png.unsqueeze(0)
        png = png.to(self.device)
        outputs = self.model(png)
        _,prediction = torch.max(outputs,1)
        predict = prediction[0].cpu().clone().numpy()
        print(predict.shape) #600,1200
        predict_ = cv2.resize(predict.astype('float'),(w,h),interpolation=cv2.INTER_NEAREST) #h,w,c

        predict_ = predict_.astype(np.uint8)
        
        predict_ = cv2.medianBlur(predict_,10)
        predict_tensor = torch.Tensor(predict_).unsqueeze(0)
        save_img(predict_tensor,'predict.png')
        predict_array = np.array(predict_tensor)
        print(np.unique(predict_array))
        seg_map = cv2.imread('predict.png')
        dst = cv2.addWeighted(frame, 0.3, seg_map, 0.7, 0)
        return dst



if __name__ == '__main__':
    INFERER = Inferer()
    frame = cv2.imread('PATH_TO_IMAGE')
    result = INFERER.predict(frame,k)
    cv2.imwrite('overlay.png',result)
    