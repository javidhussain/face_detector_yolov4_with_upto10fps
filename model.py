import cv2
import numpy as np

class Model():
    def __init__(self,model_path,cfg_path,names_path):
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.names_path = names_path
        self.classes = []
        self.Load_model()
        
    def Load_model(self):
        self.network = cv2.dnn.readNet(self.model_path,self.cfg_path)
        with open(self.names_path, "r") as names_file:
            self.classes = [line.strip() for line in names_file.readlines()]
        self.layer_names = self.network.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]
    
    def Run_model(self,img):
        self.x, self.y, self.w, self.h, self.label = 0, 0, 0, 0, 0
        self.img = img
        self.height, self.width, self.color = self.img.shape
        self.blob = cv2.dnn.blobFromImage(self.img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        self.network.setInput(self.blob)
        self.outs = self.network.forward(self.output_layers)
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        for self.out in self.outs:
            for self.detection in self.out:
                self.scores = self.detection[5:]
                self.class_id = np.argmax(self.scores)
                self.confidence = self.scores[self.class_id]
                if self.confidence > 0.5:
                    self.center_x = int(self.detection[0] * self.width)
                    self.center_y = int(self.detection[1] * self.height)
                    self.width = int(self.detection[2] * self.width)
                    self.height = int(self.detection[3] * self.height)
                    self._x = int(self.center_x - self.width / 2)
                    self._y = int(self.center_y - self.height / 2)
                    self.boxes.append([self._x, self._y, self.width, self.height])
                    self.confidences.append(float(self.confidence))
                    self.class_ids.append(self.class_id)
        self.indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)
        for i in range(len(self.boxes)):
            if i in self.indexes:
                self.x, self.y, self.w, self.h = self.boxes[i]
                self.label = str(self.classes[self.class_ids[i]])
                
                
                return [self.x,self.y,self.w,self.h,self.label]
                
                
                
                