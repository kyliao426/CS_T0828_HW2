# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:25:14 2020

@author: kuanyu
"""

import os, json, cv2, random, wget

# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


setup_logger()


url = 'http://images.cocodataset.org/val2017/000000439715.jpg'
filename = wget.download(url, "test_input.jpg")
im = cv2.imread("./test_input.jpg")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1],
               MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imshow("test", im)
cv2.imshow('res', out.get_image()[:, :, ::-1])
cv2.waitKey()
cv2.destroyAllWindows()
