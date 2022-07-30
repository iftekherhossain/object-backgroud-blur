import os
import torch
import cv2
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression,xyxy2xywh
from utils.plots import save_one_box
import torch
# Define Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__")))

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

dim = (640,640)
device = torch.device('cpu')

list_of_objects = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] 

def inference_with_bbox(model,nid_img):
    '''
    Inference Function
    '''
    scale_x=nid_img.shape[1]/dim[1]
    scale_y=nid_img.shape[0]/dim[0]
    image = cv2.resize(nid_img, dim, interpolation = cv2.INTER_AREA)
    image = image / 255 # Rescale
    image = np.moveaxis(image,2,0) # channel last -> channel first
    image = torch.from_numpy(image).to(device) # convert image to torch and move to the device
    image = image.float() # convert to double float
    image = image[None] # expand dims for batch operations
    pred = model(image)
    nms_pred = non_max_suppression(pred, 0.25, 0.45)
    class_names = []
    confidence_scores = []
    bounding_boxes = []
    for i, det in enumerate(nms_pred):
        for *xyxy, conf, cls in reversed(det):
            class_index = int(cls.numpy())
            # print(object_list[class_index])
            bounding_box = [a.numpy().tolist() for a in xyxy]
            bounding_box[0] = int(bounding_box[0]*scale_x)
            bounding_box[1] = int(bounding_box[1]*scale_y)
            bounding_box[2] = int(bounding_box[2]*scale_x)
            bounding_box[3] = int(bounding_box[3]*scale_y)
            # print(bounding_box)
            class_names.append(list_of_objects[class_index])
            bounding_boxes.append(bounding_box)
            confidence_scores.append(conf.numpy())
    return class_names, bounding_box, confidence_scores
image = cv2.imread("dog.jpg")
c,b,conf = inference_with_bbox(model,image)
print(c)