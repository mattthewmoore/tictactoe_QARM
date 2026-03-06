import cv2
from matplotlib.pyplot import gray
import numpy as np
# from pal.utilities.vision import Camera3D
# import pyrealsense2 as rs
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
    # def live_RGB(self, color):
    
    
    #     while True:
    #         ret,frame=self.cap.read()
    #         if ret:
    #             if color == 'red':
    #                 self.uppercolor=self.upper_red
    #                 self.lowercolor=self.lower_red
    #             elif color == 'blue':
    #                 self.uppercolor=self.upper_blue
    #                 self.lowercolor=self.lower_blue
    #             elif color == 'green':
    #                 self.uppercolor=self.upper_green
    #                 self.lowercolor=self.lower_green
    #             elif color == 'white':
    #                 self.uppercolor=self.upper_white
    #                 self.lowercolor=self.lower_white
                
                
    #             # print(f" upper {self.uppercolor}")
    #             # print(f" lower {self.lowercolor}")
    #             self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #             mask=cv2.inRange(self.hsv, self.lowercolor, self.uppercolor)
    #             res=cv2.bitwise_and(frame, frame, mask=mask)
    #             cv2.imshow('Live RGB Feed', res)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break    
    def choose_hsv_color(self, color):
            '''
            Allows user to change generic color to their specific color ('green', 'red', 'blue').
            Changes the objects color to that specific color
        
            Parameters:
               color (String): color of object user wants ('green', 'red', 'blue')
                
            Returns:
                None 
            '''
            
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
                    cv2.destroyAllWindows()
                    break       

    def get_depth(self):
        self.camera.read_depth(dataMode='M')
        return self.camera.imageBufferDepthM
    
   
   
    def get_depth_centroid(self,centroid_frame,color):
        '''
        Gets the depth of a list of centroids from objects of a specific color from a given frame
        
        Parameters:
            color (String): color of object user wants ('green', 'red', 'blue')
            centroid_frame (numpy array): the given frame from cap.read()
            
        Returns:
            centroid_depth_arr: a list of Z-coordinates for the centroids detected'''
        
        self.camera.read_depth(dataMode = 'M')
        depth_array = self.camera.imageBufferDepthM
        centroid_depth_arr = []
        centroid = self.create_centroid(centroid_frame,color)
        for (cx,cy) in centroid:
            depth_centroid = self.get_stable_depth_data(cx,cy, depth_array)
            centroid_depth_arr.append(depth_centroid)

        return centroid_depth_arr
    
    def get_stable_dept_data(self,cx,cy, depth_frame):
        '''
        Gets and returns the depth of an array of coordinates. It averages the data in
        a given range and takes out bad data
        
        Parameters:
            cx: The x-coordinate of the centroid
            cy: teh y-coordinate of the centroid
            depth_frame: An array of depth data for the given camera frame and resolution
        
        Returns:
            depth: A double of the depth in meters'''
        
        valid_readings = []
        depth = 0.0
        window = 5

        for i in range (-window,window+1):
            for j in range(-window, window+1):
                x = cx + i
                y = cy + j
                
                if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                    d = depth_frame[y,x]

                    if d > 0:
                        valid_readings.append(d)
        average = float(np.median(valid_readings))
        if valid_readings:
            depth = average
        return depth

    
    # def live_depth(self):
    #     while True:
    #         depth_frame = self.get_depth()

    #         depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
    #         depth_display = depth_display.astype('uint8')

    #         cv2.imshow('Depth Feed', depth_display)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break


    
    # def draw_outline(self):
    #      while True:
    #         ret1,frame=self.cap.read()
    #         key=cv2.waitKey(1) & 0xFF
    #         if ret1:
    #             imgray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             ret2,thresh= cv2.threshold(imgray,127,255,0)
    #             if ret2:
    #                 contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #                 cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    #                 cv2.imshow('Live Feed', frame)
    #         if key == ord('q'):
    #                 break 

    def get_outline(self, mask, frame):
        '''
        Returns countours numpy array of from given mask

        Args:
            mask (numpy array): array of masked frame from get_mask()
            frame (numpy array): the given frame from cap.read()
            
        Returns:
            valid_contours (list): list of big enough contours (>500 pixels)
        '''
        counter=0
        valid_contours=[]
        ret,thresh= cv2.threshold(mask,127,255,0)
        if ret:
            contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area=cv2.contourArea(c)
                if area>500:
                    # cv2.drawContours(frame, c, -1, (0,255,0), 3)
                    counter+=1
                    valid_contours.append(c)
            #print(f"contours {counter}")
            # while True:
            #     key=cv2.waitKey(1) & 0xFF
            #     cv2.imshow('mask', mask)
            #     cv2.imshow('Outline', frame)
            #     if key == ord('q'):
            return valid_contours
                
        return None
    def get_mask(self, color, frame):
        '''
        Returns mask numpy array of given color from given frame

        Args:
            color (String): color of object user wants ('green', 'red', 'blue')
            frame (numpy array): the given frame from cap.read()
            
        Returns:
            mask (numpy array): array of masked frame from cv2.inRange()
        '''
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
        self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(self.hsv, self.lowercolor, self.uppercolor)
        return mask
    # def record_outline(self,color):
    #      while True:    
    #         ret,frame=self.cap.read()
    #         key=cv2.waitKey(1) & 0xFF
    #         if ret:
    #             self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #             cv2.imshow('Live Feed', frame)
    #             if key == ord('r'):
    #                 mask=self.get_mask(color)
    #                 contour=self.get_outline(mask,frame)
    #                 mc=self.get_centoid(contour)

    #             if key == ord('q'):
    #                 break
    def get_centoid(self,contours):
        '''
        Returns centroid list of given contour list

        Args:
            contours (list): list of contours
            
        Returns:
            mc (list): list of (cx,cy) tuples of the contours 2D centroids from cv2.moments()
        '''
        mu= [cv2.moments(c) for c in contours]
        mc=[( m['m10']/m['m00'] , m['m01']/m['m00'] ) for m in mu]
        return mc
    def create_centroid(self,frame, color):
        '''
        Returns the centroid by calculating the mask, then contours then centroid.
        Uses get_mask(), get_outline(), and get_centroid()

        Args:
            color (String): color of object user wants ('green', 'red', 'blue')
            frame (numpy array): the given frame from cap.read()
        Returns:
            mc (list): list of (cx,cy) tuples of the contours 2D centroids from cv2.moments()
        '''
        mask=self.get_mask(color, frame)
        valid_contour=self.get_outline(mask, frame)
        centroid=self.get_centoid(valid_contour)
        return centroid
    def draw_contours(self, color):
        '''
        Draws the contours and centroid of the objects of the selected color live on the camera

        Args:
            color (String): color of object user wants ('green', 'red', 'blue')

        Returns:
            None
        '''
        while True:    
            ret,frame=self.cap.read()
            key=cv2.waitKey(1) & 0xFF
            if ret:
                mask=self.get_mask(color, frame)
                valid_contour=self.get_outline(mask, frame)
                centroid=self.get_centoid(valid_contour)
                self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                for c, (cx,cy) in zip(valid_contour, centroid):
                    cv2.drawContours(frame, [c], -1, (0,255,0), 3)
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.imshow('Live Outline Feed', frame)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break
                
if __name__ == "__main__":
    cam = Camera()
    cam.live_feed()