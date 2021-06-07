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
import time 

class Inferer():
    def __init__(self):
        self.device = torch.device('cuda:2')
        self.model = DeepLab(num_classes=3,
                    backbone='mobilenet',
                    output_stride=8)
        ckpt = torch.load('run/class3_add/model_best.pth')['state_dict']
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.clahe = CLAHE(p=0.8)
        self.img_transform = transforms.Compose([
                                                  transforms.Resize((500,1000)),
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
        print(predict_.shape)
        print(np.unique(predict_))
        predict_tensor = torch.Tensor(predict_).unsqueeze(0)
        save_img(predict_tensor,'Overlay/overlay/'+str(k)+'.png')
        seg_map = cv2.imread('Overlay/overlay/'+str(k)+'.png')
        dst = cv2.addWeighted(frame, 0.3, seg_map, 0.7, 0)
        return dst
    def predict_once(self,frame):
        w,h = 640,360
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
        predict_ = cv2.resize(predict.astype('float'),(w,h),interpolation=cv2.INTER_NEAREST) #h,w,c

        predict_ = predict_.astype(np.uint8)
        print(predict_.shape)
        print(np.unique(predict_))
        predict_tensor = torch.Tensor(predict_).unsqueeze(0)
        save_img(predict_tensor,'tensor.png')
        seg_map = cv2.imread('tensor.png')
        dst = cv2.addWeighted(frame, 0.3, seg_map, 0.7, 0)
        return dst



def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[0:-4]))
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


# if __name__ == '__main__':
#     INFERER = Inferer()
#     cap = cv2.VideoCapture('highway.mp4')
#     k = 0
#     while(True):
#         ret,frame = cap.read()
#         if ret:
#             start = time.time()
#             result = INFERER.predict(frame,k)
#             end = time.time()
#             result = cv2.resize(result,(640,360))
#             cv2.imwrite('Overlay/frames/'+str(k)+'.png',result)
#             print('Time:',end-start)
#             k += 1
#         if k>=5000:
#             break
#     pathIn= 'Overlay/frames/'
#     pathOut = 'video.avi'
#     fps = 30
#     convert_frames_to_video(pathIn, pathOut, fps)


if __name__ == '__main__':
    INFERER = Inferer()
    frame = cv2.imread( ## )
    start = time.time()
    result = INFERER.predict(frame)
    end = time.time()
    result = cv2.resize(result,(640,360))
    cv2.imwrite('Overlay.png',result)
    print('Time:',end-start)
