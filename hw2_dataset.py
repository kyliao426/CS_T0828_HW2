# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:52:12 2020

@author: kuanyu
"""


import os
import cv2
import h5py
import json
import random
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs


def get_dicts(img_dir):
    h5_path = os.path.join(img_dir, 'digitStruct.mat')
    h5_file = h5py.File(h5_path, 'r')
    dataset_dicts = []

    for i in range(h5_file['/digitStruct/name'].shape[0]):
    # for i in range(100):
        img_name = get_name(i, h5_file)
        img_bbox = get_bbox(i, h5_file)

        # json_file = os.path.join(img_dir, "via_region_data.json")
        # with open(json_file) as f:
        #     imgs_anns = json.load(f)
        record = {}

        filename = os.path.join(img_dir, img_name)
        height, width = cv2.imread(filename).shape[:2]
        print('processing:img{0}'.format(i))

        record["file_name"] = filename
        record["image_id"] = i
        record["height"] = height
        record["width"] = width

        objs = []
        for idx in range(len(img_bbox['label'])):
            if img_bbox['label'][idx] == 10:
                img_bbox['label'][idx] = 0
            obj = {
                "bbox": [img_bbox['left'][idx], img_bbox['top'][idx],
                         img_bbox['width'][idx], img_bbox['height'][idx]],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": int(img_bbox['label'][idx])
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


img_dir = 'HW2dataset\\train'
dataset = get_dicts(img_dir)
with open('HW2_train.json', 'w') as file_out:
    json.dump(dataset, file_out)

# with open('HW2_train.json', 'r') as file_read:
#     data_in = json.load(file_read)

# DatasetCatalog.register('trainset', lambda: myfunc(dataset))
# MetadataCatalog.get('trainset').set(thing_classes=['0', '1', '2', '3', '4',
#                                                    '5', '6', '7', '8', '9'])
# train_metadata = MetadataCatalog.get('trainset')


# for d in random.sample(dataset, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1],
#                             metadata=train_metadata,
#                             scale=2)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow("test:{0}".format(d), out.get_image()[:, :, ::-1])
# cv2.waitKey()
# cv2.destroyAllWindows()
