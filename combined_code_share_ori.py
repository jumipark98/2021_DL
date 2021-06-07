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
        self.predict_ = cv2.cvtColor(cv2.imread('predict.png'),cv2.COLOR_BGR2GRAY)
        self.dst = cv2.cvtColor(cv2.imread('overlay.png'),cv2.COLOR_BGR2GRAY)
    def predict_once(self):
        w,h = 640,480
        # h,w,c = frame.shape

        cv2.imshow('overlay',self.dst)
        self.posList = []
        cv2.namedWindow('overlay')
        cv2.moveWindow('overlay', 40, 30)
        cv2.setMouseCallback('overlay', self.onMouse)
        cv2.waitKey(0)
        start = self.posList[0]
        end  = self.posList[1]

        print(start,end)

        ans = self.A_star(start,end,self.predict_) #start=(0,3),ending=(500,300)
        print(ans)
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

    def A_star(self,start,ending,frame):
        iter = 0
        frame = frame
        map_y, map_x = 700,700
        print(map_y,map_x)
        print(frame.shape)

        self.pG=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
        self.pF=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
        self.pH=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
        self.pq = PriorityQueue()
        ans = []
        self.closed = set()

        #start = (358, 362)
        #ending = (356, 241)
        dx = (1, 1, 0, -1, -1, -1, 0, 1)
        dy = (0, 1, 1, 1, 0, -1, -1, -1)
        self.pq.put((0, start))
        now = start
        print(self.predict_.shape)
        for i in range(0, 478):
            for j in range(0, 638):
                if self.predict_[i][j] != 1:
                    self.closed.add((j, i))

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
            
        return ans
    

if __name__ == '__main__':
    INFERER = Inferer()
    start = time.time()
    INFERER.predict_once()
    end = time.time()
    print('Time:',end-start)
