import numpy as np
from queue import PriorityQueue
import cv2
import matplotlib.pyplot as plt

def A_star(start,end,frame):
    frame
    image = 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    print(image.shape)
    map_x = 500
    map_y = 640
    pG=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
    pF=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
    pH=np.array(np.zeros(map_x*map_y)).reshape(map_x,map_y)
    
    pq = PriorityQueue()
    ans=[]

    closed=set()
    start=(0,3) #start
    ending = (500,300) # end

    dx = (1,1,0,-1,-1,-1,0,1)
    dy = (0,1,1,1,0,-1,-1,-1)

    pq.put((0,start))

    now=start

    for j in range(100,200):
        for i in range(100,200):
            closed.add((j,i))

    for j in range(250,350):
        for i in range(250,450):
        closed.add((j,i))

    chk(map_x,map_y)
    x_g=[]

    y_g=[]

    while now !=ending:
        now = pq.get()[1]

        pq = PriorityQueue()

        closed.add(now) 
        print(now)

        ans.append((now[0],now[1]))

        image[now[0]][now[1]]=0
        x_g=np.append(x_g,now[0])

        y_g=np.append(y_g,now[1])
        plt.scatter(x_g,y_g)

        plt.pause(0.001)

        for i in range(0,8):

            x = now[0]+dx[i]

            y = now[1]+dy[i]

        if x<0 or y<0: continue

        if (x,y) in closed: continue

        if dx[i] ==0 or dy[i] == 0:
            g((x,y),now,10)

        else:
            g((x,y),now,14)

        dot = (x,y)

        h(dot,ending)

        F(dot)

        pq.put((pF[y][x],(x,y)))

    plt.imsave('ex.png')

def g(next, now, plus):

    pG[next[0],next[1]] = pF[now[0],now[1]]+plus

 # print(next[0],next[1])

def h(next, end):

    x = abs(end[0] - next[0])

    y = abs(end[1] - next[1])

    pH[next[0],next[1]] = (x+y)*10

def F(next):

    pF[next[0],next[1]]=pH[next[0],next[1]]+pG[next[0],next[1]]

  #print(int(pF[next[0],next[1]]))

def chk(map_x,map_y):


    for j in range(0,100):

        for i in range(0,100):

            if image[j][i]=='$':

        closed.add(j,i)