import os 
import shutil

tgtlist = ['213004','213003','213002','213001','213000','200001','200002','200003','200004','200000']

# os.mkdir('/SSD3/jumi/ASCP')
# os.mkdir('/SSD3/jumi/ASCP/Annotations2')
# os.mkdir('/SSD3/jumi/ASCP/PNGImages')

# for image in os.listdir('/SSD3/jumi/aeroscapes/JPEGImages'):
#     for tgt in tgtlist:
#         if tgt in image:
#             imagename = image[:-4]
#             shutil.copy('/SSD3/jumi/aeroscapes/JPEGImages/'+imagename+'.jpg', '/SSD3/jumi/ASCP/PNGImages/'+imagename+'.png')
# #             shutil.copy('/SSD3/jumi/aeroscapes/SegmentationClass/'+imagename+'.png','/SSD3/jumi/ASCP/Annotations/'+imagename+'.png')
# import numpy as np
# from PIL import Image 
# masklist = os.listdir('/SSD3/jumi/ASCP/Annotations/')
# # os.mkdir('/SSD3/jumi/ASCP/Annotations2')
# for maskname in masklist:
#     mask = np.asarray(Image.open('/SSD3/jumi/ASCP/Annotations/'+maskname))
#     # 0,1,2,4,5,6,7,8,9,11-> 0, #10 -> 1, #3 -> 2 (car)
#     mask = np.where(mask==1,0,mask)
#     mask = np.where(mask==2,0,mask)
#     mask = np.where(mask==4,0,mask)
#     mask = np.where(mask==5,0,mask)
#     mask = np.where(mask==6,0,mask)
#     mask = np.where(mask==7,0,mask)
#     mask = np.where(mask==8,0,mask)
#     mask = np.where(mask==9,0,mask)   
#     mask = np.where(mask==11,0,mask)
#     mask = np.where(mask==10,1,mask)
#     mask = np.where(mask==3,2,mask)
#     Image.fromarray(mask.astype('uint8')).save('/SSD3/jumi/ASCP/Annotations2/'+maskname)

os.mkdir('/SSD3/jumi/ASCP/PNGImages/train/')
os.mkdir('/SSD3/jumi/ASCP/PNGImages/val/')
os.mkdir('/SSD3/jumi/ASCP/Annotations2/train')
os.mkdir('/SSD3/jumi/ASCP/Annotations2//val')
allfilename = os.listdir('/SSD3/jumi/ASCP/PNGImages/')
allfilename = [file for file in allfilename if file.endswith(".png")]
unit = int(len(allfilename)/5)
import random 
random.shuffle(allfilename)
trainlist = allfilename[unit:]
testlist = allfilename[:unit]
for file in trainlist:
    shutil.copy('/SSD3/jumi/ASCP/PNGImages/'+file,'/SSD3/jumi/ASCP/PNGImages/train/'+file)
    shutil.copy('/SSD3/jumi/ASCP/Annotations2/'+file,'/SSD3/jumi/ASCP/Annotations2/train/'+file)
    
for file in testlist:
    shutil.copy('/SSD3/jumi/ASCP/PNGImages/'+file,'/SSD3/jumi/ASCP/PNGImages/val/'+file)
    shutil.copy('/SSD3/jumi/ASCP/Annotations2/'+file,'/SSD3/jumi/ASCP/Annotations2/val/'+file)