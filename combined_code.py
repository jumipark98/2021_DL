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
import matplotlib.pyplot as plt
from queue import PriorityQueue

###
class Inferer():
    def __init__(self):
        self.device = torch.device('cuda:0')
        self.model = DeepLab(num_classes=3,
                    backbone='mobilenet',
                    output_stride=8)
        ckpt = torch.load('run/class3_add/model_best.pth',map_location='cuda:0')['state_dict']
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.clahe = CLAHE(p=0.8)
        self.img_transform = transforms.Compose([
                                                  transforms.Resize((500,1000)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
        
       
    def predict_once(self,frame):
        w,h = 1000,500
        # h,w,c = frame.shape
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
        frame_ = cv2.resize(frame,(w,h))
        
        predict_ = predict_.astype(np.uint8)
        Image.fromarray(predict_).save('predict.png')
        not_road = np.where(predict_!=1)
        
        predict_tensor = torch.Tensor(predict_).unsqueeze(0)
        save_img(predict_tensor,'tensor.png')
        seg_map = cv2.imread('tensor.png')
        self.dst = cv2.addWeighted(frame_, 0.3, seg_map, 0.7, 0)
        cv2.imwrite('overlay.png',self.dst)
        cv2.imshow('overlay',self.dst)
        self.posList = []
        cv2.namedWindow('overlay')
        cv2.moveWindow('overlay', 40, 30)
        cv2.setMouseCallback('overlay', self.onMouse)
        cv2.waitKey(0)
        start = self.posList[0]
        end  = self.posList[1]
        print(start,end)

        ans = self.A_star(start,end,frame_,not_road) #start=(0,3),ending=(500,300)
      
    def onMouse(self,event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.posList.append((x,y))
            cv2.circle(self.dst,(x,y),10,(255,0,0),-1)

        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.destroyAllWindows()
            
    def g(self,next, now, plus):
        self.pG[next[0],next[1]] = self.pF[now[0],now[1]]+plus

    # print(next[0],next[1])

    def h(self,next, end):
        x = abs(end[0] - next[0])
        y = abs(end[1] - next[1])
        self.pH[next[0],next[1]] = (x+y)*10

    def F(self,next):
        self.pF[next[0],next[1]]=self.pH[next[0],next[1]]+self.pG[next[0],next[1]]

    def A_star(self,start,ending,frame,not_road):
        iter = 0
        frame = frame
        map_x, map_y,_ = frame.shape
        print(map_y,map_x)

        self.pG=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
        self.pF=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
        self.pH=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
        
        self.pq = PriorityQueue()
        ans = []

        self.closed=set()

        dx = (1,1,0,-1,-1,-1,0,1)
        dy = (0,1,1,1,0,-1,-1,-1)

        self.pq.put((0,start))

        now = start
        xs,ys = not_road
        for i in range(len(xs)):
            self.closed.add((ys[i],xs[i]))
        
        x_g=[]
        y_g=[]

        while now != ending:
            
            now =self.pq.get()[1]
            self.pq = PriorityQueue()
            self.closed.add(now)  #y,x
            
            ans.append((now[0],now[1])) #y,x

            for i in range(0,8):
                x = now[0]+dx[i]
                y = now[1]+dy[i]

            if x<0 or y<0: 
                print('index error')
                continue
            
            if (x,y) in self.closed:
                print('x,y in self.closed') 
                continue

            if dx[i] ==0 or dy[i] == 0:
                self.g((x,y),now,10)
            else:
                self.g((x,y),now,14)

            dot = (x,y)

            self.h(dot,ending)

            self.F(dot)

            self.pq.put((self.pF[x][y],(x,y)))
            iter += 1
            print(ans)
        return ans
    

if __name__ == '__main__':
    INFERER = Inferer()
    frame = cv2.imread('seq37_000900.png')
    start = time.time()
    INFERER.predict_once(frame)
    end = time.time()
    print('Time:',end-start)
