# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 21:10:37 2020

@author: kuanyu
"""


import os
import cv2
import json
import numpy as np
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
setup_logger()


# don't no why bbox_mode will change so change it back
def get_digits_dicts(dataset_dicts):
    for i in range(len(dataset_dicts)):
        for j in range(len(dataset_dicts[i]['annotations'])):
            dataset_dicts[i]['annotations'][j]['bbox_mode'] = BoxMode.XYWH_ABS
    return dataset_dicts


# read the trainset annotation
with open('HW2_train.json', 'r') as file_read:
    dataset = json.load(file_read)

# register the custom dataset
DatasetCatalog.register('trainset', lambda: get_digits_dicts(dataset))
MetadataCatalog.get('trainset').set(thing_classes=['0', '1', '2', '3', '4',
                                                   '5', '6', '7', '8', '9'])
train_metadata = MetadataCatalog.get('trainset')
dataset = get_digits_dicts(dataset)


# ============ train ===========
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("trainset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "model_final_280758.pkl"  # pre-trained model file location
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 50000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# ============ test ===========
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.DATASETS.TEST = ("trainset", )
predictor = DefaultPredictor(cfg)

result_list = []
test_path = 'HW2dataset\\test'
res_path = 'test_result'
for file in sorted(os.listdir(test_path), key=lambda x: int(x[: -4])):
    file_path = os.path.join(test_path, file)
    im = cv2.imread(file_path)
    print('predicting ' + file)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=train_metadata,
                   scale=2,
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    res_file = os.path.join(res_path, file)
    # cv2.imshow('test', v.get_image()[:, :, ::-1])
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite(res_file, v.get_image()[:, :, ::-1])

    row = {}  # a dictionary for one test image
    anno = outputs["instances"].to("cpu").get_fields()
    box = list(anno['pred_boxes'])
    bbox_temp = np.zeros([len(box), 4])

    # file format of submission is (y1, x1, y2, x2)
    for i in range(len(box)):
        bbox_temp[i] = box[i]
        bbox_temp[i][0], bbox_temp[i][1] = bbox_temp[i][1], bbox_temp[i][0]
        bbox_temp[i][2], bbox_temp[i][3] = bbox_temp[i][3], bbox_temp[i][2]

    anno['pred_classes'][np.where(anno['pred_classes'] == 0)] = 10
    row['bbox'] = np.round(bbox_temp).tolist()
    row['score'] = anno['scores'].tolist()
    row['label'] = anno['pred_classes'].tolist()
    result_list.append(row)

with open('309551107_075.json', 'w') as output:
    json.dump(result_list, output)
