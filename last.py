import os 
import numpy as np
import cv2
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import cv2
import json 
import time 
import matplotlib.pyplot as plt
from queue import PriorityQueue
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
        self.whole_path = []
    def predict_seg(self,frame):
        h,w = 500,1000
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
        self.predict_ = cv2.resize(predict.astype('float'),(w,h),interpolation=cv2.INTER_NEAREST).astype('uint8')
        self.frame_ = cv2.resize(frame,(w,h)).astype(np.uint8)
        self.predict_ = cv2.medianBlur(self.predict_,15)
        predict_tensor = torch.Tensor(self.predict_).unsqueeze(0)
        save_img(predict_tensor,'tensor.png')
        seg_map = cv2.imread('tensor.png')
        self.dst = cv2.addWeighted(self.frame_, 0.3, seg_map, 0.7, 0)
        return self.dst, self.predict_

    def predict_once(self,frame,first):
        self.spend_time = 0
        self.dst, self.predict_ = self.predict_seg(frame)
        
        if first == True:
            cv2.imshow('overlay',self.dst)
            self.posList = []
            cv2.namedWindow('overlay')
            cv2.moveWindow('overlay', 40, 30)
            cv2.setMouseCallback('overlay', self.onMouse)
            cv2.waitKey(0)
            self.start = self.posList[0]
            self.end = self.posList[1]
            cv2.destroyWindow('Overlay')

        previous_path = self.whole_path
        
        self.ans = self.A_star(self.predict_) #start=(0,3),ending=(500,300)
        # print(ans)
        for prev_points in previous_path:
            cv2.circle(self.dst, tuple(prev_points),1,(250,244,212))
        for point in self.ans:
            cv2.circle(self.dst,tuple(point),1,(204,183,61)) #B,G,R

        return self.dst

    def onMouse(self,event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.posList.append((x,y))
            cv2.circle(self.dst,(x,y),3,(255,0,0),-1) 

        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.destroyAllWindows()
            
    def g(self,next, now, plus):
        self.pG[next[0],next[1]] = self.pF[now[0],now[1]]+plus

    def h(self,next, end):
        x = abs(end[0] - next[0])
        y = abs(end[1] - next[1])
        self.pH[next[0],next[1]] = (x+y)*15

    def F(self,next):
        self.pF[next[0],next[1]]=self.pH[next[0],next[1]]+self.pG[next[0],next[1]]

    def A_star(self,frame):
        frame = frame
        map_y, map_x = 500,1000

        self.pG=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
        self.pF=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
        self.pH=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
        self.pq = PriorityQueue()
        self.ans = []
        self.closed = set()

        dx = (1, 1, 0, -1, -1, -1, 0, 1)
        dy = (0, 1, 1, 1, 0, -1, -1, -1)
        self.pq.put((0, self.start))
        now = self.start
        not_road = np.where(self.predict_!=1)
        ys,xs = not_road
        
        for i in range(len(xs)):
            self.closed.add((xs[i],ys[i]))
        
        start_time = time.time()
        iter = 0
        self.finish = False
        while now != self.end:
            print(iter)
            if iter>5:
                self.start = now
                self.end = self.end
                print("New start",self.start)
                self.whole_path.extend(self.ans)
                break
            now =self.pq.get()[1]
            self.pq = PriorityQueue()
            self.closed.add(now)  #y,x
            
            self.ans.append((now[0],now[1])) #y,x

            for i in range(0,8):
                x = now[0]+dx[i]
                y = now[1]+dy[i]

                if x<0 or y<0:
                    print('Invalid coords')
                    continue
                if (x,y) in self.closed:
                    print('x,y not in road')
                    continue
                if dx[i] ==0 or dy[i] == 0:
                    self.g((x,y),now,10)
                else:
                    self.g((x,y),now,14)

                dot = (x,y)

                self.h(dot,self.end)

                self.F(dot)

                self.pq.put((self.pF[x][y],(x,y)))

            end_time = time.time()
            self.spend_time = end_time-start_time
            iter+=1 

        self.finish == True
        
        return self.ans
    

if __name__ == '__main__':
    INFERER = Inferer()
    start = time.time()
    idx = 0
    files = os.listdir('../newframes')
    files.sort(key = lambda x: int(x[0:-4]))
    # os.mkdir('../newframes_result5')
    for file in files[1932:]:
        print(file)
        tgt = '../newframes/'+file
        frame = cv2.imread(tgt)
        if idx == 0:
            result = INFERER.predict_once(frame,True)
        else:
            result = INFERER.predict_once(frame,False)
        idx += 1
        if INFERER.finish == True:
            break
        # cv2.imshow('Result'+file,result)
        cv2.imwrite('../newframes_result5/'+file, result)
    end = time.time()
    print('Time:',end-start)
