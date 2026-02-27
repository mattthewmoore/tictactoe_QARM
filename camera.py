import cv2
import numpy as np
from pal.utilities.vision import Camera3D
# init and live feed helped by claude

class Camera:
    def __init__(self, index=1, backend=cv2.CAP_DSHOW):
        self.cap = cv2.VideoCapture(index, backend)
        self.camera = Camera3D(
            mode='RGB&DEPTH',
            frameWidthRGB=640,
            frameHeightRGB=480,
            frameWidthDepth=640,
            frameHeightDepth=480,
            deviceId='0',
            readMode=0
        )
        self.lower_red=np.array([0, 120, 70])
        self.upper_red=np.array([10, 255, 255])
        self.lower_blue=np.array([100, 100, 20])
        self.upper_blue=np.array([130, 255, 255])
        self.lower_green=np.array([40, 50, 50])
        self.upper_green=np.array([80, 255, 255])
        self.hsv=None
        
        print("Camera initialized.")

    def live_feed(self):
        cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
        while True:
           

            # Read a frame from the camera
            ret, frame = cap.read()

            #display if succesfully read
            if ret:
                cv2.imshow('Live Feed', frame)

           
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    def live_RGB(self, color):
    
        cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    
        while True:
            ret,frame=cap.read()
            if ret:
                if color == 'red':
                    self.uppercolor=self.upper_red
                    self.lowercolor=self.lower_red
                elif color == 'blue':
                    self.uppercolor=self.upper_blue
                    self.lowercolor=self.lower_blue
                elif color == 'green':
                    self.uppercolor=self.upper_green
                    self.lowercolor=self.lower_green
                print(f" upper {self.uppercolor}")
                print(f" lower {self.lowercolor}")
                self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask=cv2.inRange(self.hsv, self.lowercolor, self.uppercolor)
                res=cv2.bitwise_and(frame, frame, mask=mask)
                cv2.imshow('Live RGB Feed', res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    
    def choose_hsv_color(self, color):
            counter=0
            cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
            while True:
                counter+=1
                ret,frame=cap.read()
                key=cv2.waitKey(1) & 0xFF
                
                if ret:
                     self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                     cv2.imshow('Live Feed', frame)
                     if counter==30:
                        print(self.hsv[240][320])
                        counter=0
                if key == ord('r'):
                    if color == 'red':
                        self.lower_red=np.array([self.hsv[240][320][0] -5, 100, 20])
                        self.upper_red=np.array([self.hsv[240][320][0] +15, 255, 255])
                    elif color == 'blue':
                        self.lower_blue=np.array([self.hsv[240][320][0] -15, 100, 20])
                        self.upper_blue=np.array([self.hsv[240][320][0] +15, 255, 255])
                        
                    elif color == 'green':
                        self.lower_green=np.array([self.hsv[240][320][0] -4, 100, 20])
                        self.upper_green=np.array([self.hsv[240][320][0] +4, 255, 255])
                        print(f" lower green{self.lower_green}")
                        print(f" upper green{self.upper_green}")
                    print('recorded')
                    
                if key == ord('q'):
                    break       

    def get_depth(self):
        self.camera.read_depth(dataMode='M')
        return self.camera.imageBufferDepthM
                
    
    def live_depth(self):
        while True:
            depth_frame = self.get_depth()

            depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype('uint8')

            cv2.imshow('Depth Feed', depth_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    


            
        