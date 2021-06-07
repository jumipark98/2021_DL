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


class Inferer():
    def __init__(self):
        self.predict_ = np.asarray(Image.open('predict.png'))
        self.dst = cv2.imread('overlay.png')

    def predict_once(self):
        w,h = 640,480
        # h,w,c = frame.shape
        not_road = np.where(self.predict_!=1)
        cv2.imshow('overlay',self.dst)
        self.posList = []
        cv2.namedWindow('overlay')
        cv2.moveWindow('overlay', 40, 30)
        cv2.setMouseCallback('overlay', self.onMouse)
        cv2.waitKey(0)
        start = self.posList[0]
        end = self.posList[1]

        print(start,end)

        ans = self.A_star(start,end,self.predict_,not_road) #start=(0,3),ending=(500,300)
        print('Final Answer : ', ans)
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
        map_y, map_x = 700,700
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
        # print(self.closed[0])
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
                    
                    continue
                
                if (x,y) in self.closed:
                    
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
    start = time.time()
    INFERER.predict_once()
    end = time.time()
    print('Time:',end-start)
