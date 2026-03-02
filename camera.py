import cv2
from matplotlib.pyplot import gray
import numpy as np
# from pal.utilities.vision import Camera3D
# init and live feed helped by claude

class Camera:
    def __init__(self, index=1, backend=cv2.CAP_DSHOW):
        self.cap = cv2.VideoCapture(index, backend)
        # self.camera = Camera3D(
        #     mode='RGB&DEPTH',
        #     frameWidthRGB=640,
        #     frameHeightRGB=480,
        #     frameWidthDepth=640,
        #     frameHeightDepth=480,
        #     deviceId='0',
        #     readMode=0
        # )
        self.lower_red=np.array([0, 120, 70])
        self.upper_red=np.array([10, 255, 255])
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])
        self.lower_blue=np.array([100, 100, 20])
        self.upper_blue=np.array([130, 255, 255])
        self.lower_green=np.array([40, 50, 50])
        self.upper_green=np.array([80, 255, 255])
        self.hsv=None
        
        print("Camera initialized.")

    def live_feed(self):
        
        while True:
           

            # Read a frame from the camera
            ret, frame = self.cap.read()

            #display if succesfully read
            if ret:
                cv2.imshow('Live Feed', frame)

           
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    def live_RGB(self, color):
    
    
        while True:
            ret,frame=self.cap.read()
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
                elif color == 'white':
                    self.uppercolor=self.upper_white
                    self.lowercolor=self.lower_white
                
                
                # print(f" upper {self.uppercolor}")
                # print(f" lower {self.lowercolor}")
                self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask=cv2.inRange(self.hsv, self.lowercolor, self.uppercolor)
                res=cv2.bitwise_and(frame, frame, mask=mask)
                cv2.imshow('Live RGB Feed', res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    
    def choose_hsv_color(self, color):
            counter=0
        
            while True:
                counter+=1
                ret,frame=self.cap.read()
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

    # def get_depth(self):
    #     self.camera.read_depth(dataMode='M')
    #     return self.camera.imageBufferDepthM
                
    
    # def live_depth(self):
    #     while True:
    #         depth_frame = self.get_depth()

    #         depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
    #         depth_display = depth_display.astype('uint8')

    #         cv2.imshow('Depth Feed', depth_display)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    def draw_outline(self):
         while True:
            ret1,frame=self.cap.read()
            key=cv2.waitKey(1) & 0xFF
            if ret1:
                imgray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret2,thresh= cv2.threshold(imgray,127,255,0)
                if ret2:
                    contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
                    cv2.imshow('Live Feed', frame)
            if key == ord('q'):
                    break 

    def get_outline(self, mask, frame):
        counter=0
        ret,thresh= cv2.threshold(mask,127,255,0)
        if ret:
            contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area=cv2.contourArea(c)
                if area>500:
                    cv2.drawContours(frame, c, -1, (0,255,0), 3)
                    counter+=1
            print(f"contours {counter}")
            while True:
                key=cv2.waitKey(1) & 0xFF
                cv2.imshow('mask', mask)
                cv2.imshow('Outline', frame)
                if key == ord('q'):
                    break
    def get_mask(self, color):
        ret,frame=self.cap.read()
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
            self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(self.hsv, self.lowercolor, self.uppercolor)
            return mask
    def record_outline(self,color):
         while True:    
            ret,frame=self.cap.read()
            key=cv2.waitKey(1) & 0xFF
            if ret:
                self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                cv2.imshow('Live Feed', frame)
                if key == ord('r'):
                    mask=self.get_mask(color)
                    self.get_outline(mask,frame)
                if key == ord('q'):
                    break
    


            
        