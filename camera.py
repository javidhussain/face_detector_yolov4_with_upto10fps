import cv2
import numpy as np
import time
from model import Model


class Camera():
    def __init__(self,camera,model_path,cfg_path,names_path):
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.names_path = names_path
        self.camera = camera
        self.model = Model(self.model_path,self.cfg_path,self.names_path)
        self.video_cap = cv2.VideoCapture(self.camera)
        self.success,self.frame = self.video_cap.read()
        
    
    def Run_cam(self):
        count = 0
        while True:
            self.success,self.frame = self.video_cap.read()
            if self.success != True:
                break
            self.frame = cv2.resize(self.frame,(400,400))
            
            if count % 60 == 0:
                if (self.model.Run_model(self.frame)) == None:
                    x,y,w,h = 0, 0, 0, 0
                    self.label = ""
                else:
                    x,y,w,h,self.label = self.model.Run_model(self.frame)
            
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.frame, self.label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            count += 1
            cv2.imshow("Detector",self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_cap.release()
        cv2.destroyAllWindows()
        
        