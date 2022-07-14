import cv2
import numpy as np
import time
import sys
from model import Model

class Cam_tracker():
    def __init__(self,camera,model_path,cfg_path,names_path):
        self.x,self.y,self.w,self.h = (0, 0, 0, 0)
        self.label = ""
        self.tracker = cv2.TrackerCSRT_create()
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.names_path = names_path
        self.camera = camera
        self.model = Model(self.model_path,self.cfg_path,self.names_path)
        self.video_cap = cv2.VideoCapture(self.camera)
        while True:
            self.success,self.frame = self.video_cap.read()
            start = time.time()
            self.frame = cv2.resize(self.frame,(200,200))
            print ("::{Shot Taken}::")
            if self.model.Run_model(self.frame) != None:
                print ("::{Please Wait}::")
                self.x,self.y,self.w,self.h,self.label = self.model.Run_model(self.frame)
            if (self.x !=0 or self.y !=0 or self.w !=0 or self.h !=0) and (self.label == "face"):
                self.width = (self.x+self.w)
                self.height = (self.y+self.h)
                self.bbox = (self.x,self.y,self.width,self.height)
                self.success = self.tracker.init(self.frame, self.bbox)
                fps = (time.time()-start)
                print ("::{Face Detected}::",self.x,self.y,self.width,self.height)
                #self.video_cap.release()
                break
            else:
                print ("::{Face NOT Detected}::")
                
        
    def Run_tracker_cam(self):
        #count = 0
        while True:
            self.success,self.frame = self.video_cap.read()
            self.start_time = time.time()
            if self.success != True:
                break
            self.frame = cv2.resize(self.frame,(200,200))
            self.success,self.bbox_traker = self.tracker.update(self.frame)
            if self.success:
                self.p1 = (int(self.bbox_traker[0]) , int(self.bbox_traker[1]) )
                self.p2 = (int(self.bbox_traker[0] + self.bbox_traker[2]), int(self.bbox_traker[1] + self.bbox_traker[3]))
                cv2.rectangle(self.frame, self.p1, self.p2, (255,0,0), 2, 1)
                cv2.putText(self.frame, self.label, self.p1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            else:
                print ("::{No Face Found}::")
                self.video_cap.release()
                cv2.destroyAllWindows()
                break
            
            self.end_time = time.time()
            self.fps = 1/(self.end_time - self.start_time)
            cv2.putText(self.frame, "FPS : " +str(float(self.fps)), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,250), 1)
            cv2.imshow("Detector",self.frame)
            #time.sleep(0.060)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_cap.release()
        cv2.destroyAllWindows()
        sys.exit()
    def Release(self):
        self.video_cap.release()
        cv2.destroyAllWindows()
            
            
        
        
            
        