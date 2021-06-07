import os 
import numpy as np
from PIL import Image 

phases = ['train','val']
temp_list = []
for phase in phases:
    ann_path = '/SSD3/jumi/ASCP/Annotations2/'+phase
    for ann in os.listdir(ann_path):
        tgt = np.array(Image.open(ann_path+'/'+ann))
        temp_list.append(tgt.max())

print(set(temp_list))

