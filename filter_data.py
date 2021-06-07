import os 
import shutil 
from PIL import Image
import numpy as np
import json 
import cv2

def filter_data(phase):
    whole_datapath = '/SSD3/jumi/SYNTHiA/'+phase.lower()
    scene_id = 0
    rec_scene = []
    for scene in os.listdir(whole_datapath):
        if ".mp4" in scene or ".txt" in scene:
            continue
        sub_scene = os.listdir(os.path.join(whole_datapath,scene))
        sub_scene_id = 0
        for sub in sub_scene:
            image_id = 0
            rgb_path = os.path.join(whole_datapath,scene,sub,'RGB')
            seg_path = os.path.join(whole_datapath,scene,sub,'SemSeg')
            for img in os.listdir(rgb_path):
                new_name = str(scene_id)+'_'+str(sub_scene_id)+'_'+str(image_id)+'_'+img
                shutil.copy(os.path.join(rgb_path,img),os.path.join('/SSD3/jumi/SYNTHiA/data/my_dataset/img_dir',phase.lower(),new_name))
                annot_c = np.array(Image.open(os.path.join(seg_path,img)))[:,:,0]
                Image.fromarray(annot_c).save(os.path.join('/SSD3/jumi/SYNTHiA/data/my_dataset/ann_dir',phase.lower(),new_name))
                image_id+=1

            scene_dict = {"ori_name":os.path.join(scene,sub),"new_name":str(scene_id)+'_'+str(sub_scene_id)}
            rec_scene.append(scene_dict)
            sub_scene_id+=1
        scene_id+=1
    with open('/SSD3/jumi/SYNTHiA/data/my_dataset/info/'+phase+'.json','w') as js:
        json.dump(rec_scene,js)

    

if __name__ == '__main__':
    phases = ['Train','Val']
    for phase in phases:
        print(phase)
        filter_data(phase)
    print('Train_img : ',len(os.listdir('/SSD3/jumi/SYNTHiA/data/my_dataset/img_dir/train')))
    print('Train_ann : ',len(os.listdir('/SSD3/jumi/SYNTHiA/data/my_dataset/ann_dir/train')))
    print('Val_img : ',len(os.listdir('/SSD3/jumi/SYNTHiA/data/my_dataset/img_dir/val')))
    print('Val_img : ',len(os.listdir('/SSD3/jumi/SYNTHiA/data/my_dataset/ann_dir/val')))
