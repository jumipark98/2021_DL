import cv2 
import os 
# os.mkdir('../newframes')
cap = cv2.VideoCapture('../newvideo.mp4')
iter = 0
while cap.isOpened():
    ret,frame = cap.read()
    if ret == True:
        cv2.imwrite('../newframes/'+str(iter)+'.png',frame)
        iter += 1

