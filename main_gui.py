import re
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog 
import cv2
import scipy as sp
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression,xyxy2xywh
from utils.plots import save_one_box
import requests
import base64
from PIL import Image
import io

class Ui_MainWindow(object):
    def __init__(self):
        self.list_of_objects = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

        self.dim = (640,640)
        self.device = torch.device('cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(718, 521)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(230, 10, 481, 471))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("hello.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.browse_button = QtWidgets.QPushButton(self.centralwidget)
        self.browse_button.setGeometry(QtCore.QRect(10, 10, 89, 25))
        self.browse_button.setObjectName("browse_button")
        self.browse_button.clicked.connect(self.clicked_browse)
        self.object_list = QtWidgets.QComboBox(self.centralwidget)
        self.object_list.setGeometry(QtCore.QRect(10, 80, 86, 25))
        self.object_list.setObjectName("object_list")
        self.individual = QtWidgets.QComboBox(self.centralwidget)
        self.individual.setGeometry(QtCore.QRect(120, 120, 86, 25))
        self.individual.setObjectName("individual")
        self.select_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_button.setGeometry(QtCore.QRect(10, 120, 89, 25))
        self.select_button.setObjectName("select_button")
        self.select_button.clicked.connect(self.clicked_select)
        self.submit_button = QtWidgets.QPushButton(self.centralwidget)
        self.submit_button.setGeometry(QtCore.QRect(120, 80, 89, 25))
        self.submit_button.setObjectName("submit_button")
        self.submit_button.clicked.connect(self.clicked_submit)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 718, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "NybBlur"))
        self.browse_button.setText(_translate("MainWindow", "Open Image"))
        self.select_button.setText(_translate("MainWindow", "Select Obj"))
        self.submit_button.setText(_translate("MainWindow", "Submit"))

    def clicked_browse(self):
        # print('clicked')
        self.fname = QFileDialog.getOpenFileName(MainWindow, "opne file", "", "All Files (*) ;; Image File (*.jpg) ;;")
        print(self.fname[0])
        self.image = cv2.imread(self.fname[0])
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(self.fname[0]))
        self.label.setScaledContents(True)
        self.object_list.clear()
        self.individual.clear()
        self.class_names, self.bboxs, self.confs = self.inference_with_bbox(self.model,self.image)
        self.object_bbox_dict = dict()
        self.different_classes = list(set(self.class_names))

        for i, cls in enumerate(self.different_classes):
            self.object_list.addItem(cls)
        
        for i,diff_class in enumerate(self.different_classes):
            self.object_bbox_dict[diff_class] = []
            for cls, bbox in zip(self.class_names, self.bboxs):
                if cls == diff_class:
                    self.object_bbox_dict[diff_class].append(bbox)
        
        print(self.object_bbox_dict)

        if len(self.class_names) > 0:
            for c_name, bbox in zip(self.class_names, self.bboxs):
                print(bbox)
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]
                # bbox_image =cv2.rectangle(self.image, (x1,y1), (x2,y2), (255,0,0), 2)

            # cv2.imwrite("bbox_image.jpg",bbox_image)
            # self.label.setPixmap(QtGui.QPixmap("temp.jpg"))

    def clicked_select(self):
        # print("Selected")
        self.individual.clear()
        self.selected_obj = self.object_list.currentText()
        individual_object_bboxes = self.object_bbox_dict[self.selected_obj]

        for i in range(len(individual_object_bboxes)):
            self.individual.addItem(f'{self.selected_obj}-{i+1}')

    def clicked_submit(self):
        # current_selection = self.object_list.currentText()
        # self.rotated_image = cv2.rotate(self.image, cv2.ROTATE_180)
        # self.rotated_image = cv2.putText(self.rotated_image, str(current_selection), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3, cv2.LINE_AA)
        # cv2.imwrite("temp.jpg",self.rotated_image)
        # self.label.setPixmap(QtGui.QPixmap("temp.jpg"))
        # print(self.individual.currentText())
        _, imagebytes = cv2.imencode(".jpg", self.image)
        temp_b64 = base64.b64encode(imagebytes).decode("utf8")
        specific = self.individual.currentText()
        specific_li = specific.split("-")
        bbox = self.object_bbox_dict[specific_li[0]][int(specific_li[1])-1]
        # print(self.object_bbox_dict[specific_li[0]])
        # print(bbox)
        form_data = {
            'image': temp_b64,
            'bbox' : bbox  
            }
        # print(self.image.shape)
        response = requests.post('http://0.0.0.0:8000/get_blur/',data=form_data)
        # print(response.text)
        final_image = np.asarray(Image.open(io.BytesIO(base64.b64decode(response.text))))
        # print(final_image)
        stat = cv2.imwrite("final_image.jpg",final_image)
        self.label.setPixmap(QtGui.QPixmap("final_image.jpg"))

    def inference_with_bbox(self,model,img):
        '''
        Inference Function
        '''
        scale_x=img.shape[1]/self.dim[1]
        scale_y=img.shape[0]/self.dim[0]
        image = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        image = image / 255 # Rescale
        image = np.moveaxis(image,2,0) # channel last -> channel first
        image = torch.from_numpy(image).to(self.device) # convert image to torch and move to the device
        image = image.float() # convert to double float
        image = image[None] # expand dims for batch operations
        pred = model(image) # Prediction
        nms_pred = non_max_suppression(pred, 0.25, 0.45) #NMS
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
                class_names.append(self.list_of_objects[class_index])
                bounding_boxes.append(bounding_box)
                confidence_scores.append(conf.numpy())
        return class_names, bounding_boxes, confidence_scores

    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
