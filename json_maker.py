from __future__ import division
from __future__ import print_function

import os
import cv2
import json, yaml
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm


info = {"year" : 2021,
        "version" : "1.0",
        "description" : "nakdongage",
        "contributor" : "Junseok ",
        "url" : "http://",
        "date_created" : "2021"}
licenses = [{"id": 1,
            "name": "Attribution-NonCommercial",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }]

_type = "instances"

categories = [{"id": 1, "name":"boat", "supercategory": "boat"}]



def __get_annotation_image_pairs():
    images = []
    annotations = []
    image_set = glob.glob("./val2017/*.jpg")
    image_id = 0
    anno_id = 0
    for image in tqdm(image_set):
    
        data_name = image.split("/")[-1].split(".")[0]

        img = cv2.imread("{}".format(image))

        images.append({"date_captured" : "2021",
                        "file_name" : data_name+".jpg", # remove "/"
                        "id" : image_id+1,
                        "license" : 1,
                        "url" : "",
                        "height" : img.shape[0],
                        "width" : img.shape[1]})
     
        f = open("./label/{}.txt".format(data_name), 'r')
        lines = f.readlines()
        for line in lines:
            line_info = line.split(" ")
            category = int(line_info[0])
            if category == 0:
                category = 10
            x = float(line_info[1])
            y = float(line_info[2])
            w = float(line_info[3])
            h = float(line_info[4])

            bbox = [x,y,w,h]

            annotations.append({"segmentation" : [0.0, 3.0, 5.0, 7.0, 9.0, 10.0],
                                "area" : 10.0,
                                "iscrowd" : 0,
                                "image_id" : image_id+1,
                                "bbox" : bbox,
                                "category_id" : category,
                                "id": anno_id+1})
            anno_id += 1
      
        f.close()
        image_id += 1
    return images, annotations
images, annotations = __get_annotation_image_pairs()

json_data = {"info" : info,
            "images" : images,
            "licenses" : licenses,
            "type" : _type,
            "annotations" : annotations,
            "categories" : categories}

with open("./annotations/instances_val2017.json", "w") as jsonfile:
    json.dump(json_data, jsonfile, sort_keys=True, indent=4)