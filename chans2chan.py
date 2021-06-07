color2index = {
    (0,0,0) : 0,
    (128,0,0) : 1,
    (128,64,128) : 2,
    (0,128,0) : 3,
    (128,128,0) : 4,
    (64,0,128) : 5,
    (192,0,192) : 6,
    (64,64,0) : 7
}
# Background clutter       	(0,0,0)
# Building			(128,0,0)
# Road				(128,64,128)
# Tree				(0,128,0)
# Low vegetation		        (128,128,0)
# Moving car			(64,0,128)
# Static car			(192,0,192)
# Human				(64,64,0)
def rgb2mask(img):

    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0],[1],[2]])

    img_id = img.dot(W).squeeze(-1) 
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        try:
            mask[img_id==c] = color2index[tuple(img[img_id==c][0])] 
        except:
            pass
    return mask


import os 
import shutil
from PIL import Image
import cv2
import numpy as np

# os.mkdir('/SSD3/jumi/UAVID/PNGImages/')
# os.mkdir('/SSD3/jumi/UAVID/Annotations/')

# os.mkdir('/SSD3/jumi/UAVID/PNGImages/train')
# os.mkdir('/SSD3/jumi/UAVID/Annotations/val')

# os.mkdir('/SSD3/jumi/UAVID/PNGImages/val')
# os.mkdir('/SSD3/jumi/UAVID/Annotations/train')

# phases = ['train','val']
# for phase in phases:
#     tgtfolder = '/SSD3/jumi/UAVID/uavid_'+phase
#     for tgt in os.listdir(tgtfolder): #tgt:seq1
#         scenes = os.path.join(tgtfolder,tgt)
#         #/SSD3/jumi/UAVID/uavid_train/seq1/Images+Labels
#         for scene in os.listdir(os.path.join(scenes,'Labels')): #scene:00001.png
#             rgbmask = np.array(Image.open(os.path.join(scenes,"Labels",scene)))
#             mask = rgb2mask(rgbmask).astype('uint8')
#             newname = tgt+'_'+scene
#             shutil.copy(os.path.join(scenes,'Images',scene),os.path.join('/SSD3/jumi/UAVID/PNGImages/',phase,newname))
#             Image.fromarray(mask).save(os.path.join('/SSD3/jumi/UAVID/Annotations/',phase,newname))

os.mkdir('/SSD3/jumi/UAVID/Annotations2')
os.mkdir('/SSD3/jumi/UAVID/Annotations2/val')
os.mkdir('/SSD3/jumi/UAVID/Annotations2/train')

phases = ['train','val']
for phase in phases:
    tgtfolder = '/SSD3/jumi/UAVID/Annotations/'+phase
    for tgt in os.listdir(tgtfolder):
        ori = np.asarray(Image.open(os.path.join(tgtfolder,tgt)))
        #0,1,3,4,7 -> 0, #2 -> 1, #5,6 -> 2
        temp_ori = ori
        new_ori = np.where(ori==2,100,ori)
        new_ori = np.where(new_ori==5,101,new_ori)
        new_ori = np.where(new_ori==6,101,new_ori)
        new_ori = np.where(new_ori==1,0,new_ori)
        new_ori = np.where(new_ori==3,0,new_ori)
        new_ori = np.where(new_ori==4,0,new_ori)
        new_ori = np.where(new_ori==7,0,new_ori)
        new_ori = np.where(new_ori==100,1,new_ori)
        new_ori = np.where(new_ori==101,2,new_ori)
        
        Image.fromarray(new_ori.astype('uint8')).save(os.path.join('/SSD3/jumi/UAVID/Annotations2/',phase,tgt))