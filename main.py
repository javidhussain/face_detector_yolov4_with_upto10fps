from camera import Camera
import time
from camera_tracker import Cam_tracker

cam = -1
path_to_model = "models/crowdhuman_416_11000.weights"
path_to_cfg   = "models/crowdhuman_416.cfg"
path_to_names = "models/crowdhuman.names"


while True:
    video = Cam_tracker(cam,path_to_model,path_to_cfg,path_to_names)
    # time.sleep(2)
    if 0xff == 27 :
        video.Release()
        break
    # video.Run_tracker_cam()
    # time.sleep(1)
    if 0xff == 27 :
        video.Release()
        break