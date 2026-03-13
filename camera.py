import cv2
from matplotlib.pyplot import gray
import numpy as np
import pyrealsense2 as rs
# init and live feed helped bys claude

class Camera:
    def __init__(self, index=1, backend=cv2.CAP_DSHOW):
        self.cap = cv2.VideoCapture(index, backend)
        self.lower_red=np.array([0, 120, 70])
        self.upper_red=np.array([10, 255, 255])
        self.lower_white = np.array([0, 0, 100])
        self.upper_white = np.array([180, 40, 180])
        self.lower_grey = np.array([88, 5, 115])
        self.upper_grey = np.array([108, 30, 100])
        self.lower_blue=np.array([95, 100, 100])
        self.upper_blue=np.array([112, 255, 255])
        self.lower_green=np.array([75 ,60, 50])
        self.upper_green=np.array([88, 255, 190])
        self.lower_gold=np.array([10 ,50, 150])
        self.upper_gold=np.array([35, 255, 255])
        self.hsv=None
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640,480,rs.format.z16,30)
        self.pipeline.start(config)

        print("Camera initialized.")

    def live_feed(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('Live Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
                    print(f" lower green{self.lower_blue}")
                    print(f" upper green{self.upper_blue}")
                elif color == 'green':
                    self.lower_green=np.array([self.hsv[240][320][0] -4, 100, 20])
                    self.upper_green=np.array([self.hsv[240][320][0] +4, 255, 255])
                elif color == 'grey':
                    self.lower_grey=np.array([self.hsv[240][320][0] -2, 100, 20])
                    self.upper_grey=np.array([self.hsv[240][320][0] +2, 255, 255])
                    print(f" lower green{self.lower_grey}")
                    print(f" upper green{self.upper_grey}")
                print('recorded')

            if key == ord('q'):
                cv2.destroyAllWindows()
                break

    def get_depth_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_frame, depth_image

    def get_distance(self, depth_frame, cx, xy):
        return depth_frame.get_distance(cx,cy)

    def live_depth_feed(self):
        while True:
            depth_frame, depth_image = self.get_depth_frame()
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4),
                                            cv2.COLORMAP_JET)
            cv2.imshow('Depth Feed', depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
                if area>15:
                    counter+=1
                    valid_contours.append(c)
            # print(f"contours {counter}")
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
        elif color == 'gold':
            self.uppercolor=self.upper_gold
            self.lowercolor=self.lower_gold
        elif color == 'blue':
            self.uppercolor=self.upper_blue
            self.lowercolor=self.lower_blue
        elif color == 'green':
            self.uppercolor=self.upper_green
            self.lowercolor=self.lower_green
        elif color == 'white':
            self.uppercolor=self.upper_white
            self.lowercolor=self.lower_white
        elif color == 'grey':
            self.uppercolor=self.upper_grey
            self.lowercolor=self.lower_grey
        self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(self.hsv, self.lowercolor, self.uppercolor)
        return mask

    def get_centoid(self, contours):
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

    def create_centroid(self, frame, color):
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

    # def draw_contours(self, color, depth):
    #     '''
    #     Draws the contours and centroid of the objects of the selected color live on the camera

    #     Args:
    #         color (String): color of object user wants ('green', 'red', 'blue')

    #     Returns:
    #         None
    #     '''
    #     ret,frame=self.cap.read()
    #     key=cv2.waitKey(1) & 0xFF
    #     if ret:
    #         mask=self.get_mask(color, frame)
    #         valid_contour=self.get_outline(mask, frame)
    #         true_contour=self.get_true_contour(depth, valid_contour)
    #         centroid=self.get_centoid(true_contour)
    #         self.hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #         for c, (cx,cy) in zip(true_contour, centroid):
    #             cv2.drawContours(frame, [c], -1, (0,255,0), 3)
    #             cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    #         cv2.imshow('Live Outline Feed', frame)
    #         if key == ord('q'):
    #             cv2.destroyAllWindows()
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
                #     cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.imshow('Live Outline Feed', frame)
                
                        
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break
    def init_frame(self):
        self.get

    def get_true_contour(self, depth, contours):
        results=[]
        for cnt in contours:
            color_mask=np.zeros(depth.shape,dtype=np.uint8)
            cv2.drawContours(color_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            contour_depth=np.where(color_mask>0, depth, 0)
            nomalize_contour_depth=cv2.normalize(contour_depth,None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            edges=cv2.Canny(nomalize_contour_depth, 120, 150)
            kernel = np.ones((3,3), np.uint8)
            edges= cv2.dilate(edges, kernel, iterations=2)
            subtracted = cv2.subtract(color_mask, edges)
            new_contour, hierarchy= cv2.findContours(subtracted,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in new_contour:
                area=cv2.contourArea(c)
                if area>1:
                    results.append(c)
        return results
    def order_centroid(self, centroid):
        smallest=0
        largest=0
        top_row=250
        middle_row=100
        low_row=0
        top_row_list=[]
        middle_row_list=[]
        low_row_list=[]
        if centroid is None:
            return np.zeros((9,2))
        for (cx,cy) in centroid:
            if cy>top_row:
                top_row_list.append([cx,cy])
            elif cy>middle_row:
                middle_row_list.append([cx,cy])
            else:
                low_row_list.append([cx,cy])

        top_row_list=np.array(top_row_list)
        middle_row_list=np.array(middle_row_list)    
        low_row_list=np.array(low_row_list)    
        if len(top_row_list)>0:
            top_row_list=top_row_list[np.argsort(top_row_list[:,0])]
        else:
            top_row_list=np.zeros((3,2))
        if len(middle_row_list)>0:
            middle_row_list=middle_row_list[np.argsort(middle_row_list[:,0])]   
        else:
            middle_row_list=np.zeros((3,2))
        if len(low_row_list)>0:
            low_row_list=low_row_list[np.argsort(low_row_list[:,0])]    
        else:
            low_row_list=np.zeros((3,2))

        return np.vstack((low_row_list, middle_row_list,top_row_list))
    def stable_frame_centroids(self, color, stable_frames=10, stable_zeros=10):
        '''
        Draws the contours and centroid of the objects of the selected color live on the camera

        Args:
            color (String): color of object user wants ('green', 'red', 'blue')

        Returns:
            None
        '''
        stable=0
        zeros=0
        counter=None
        while True:
            ret,frame=self.cap.read()
            if ret:
                mask=self.get_mask(color, frame)
                valid_contour=self.get_outline(mask, frame)
                if valid_contour is not None:
                    centroid=self.get_centoid(valid_contour)
                    if len(centroid)==0:
                        zeros+=1
                        stable=0
                        counter=None
                        if zeros==stable_zeros:
                            # print("No contours found")
                            return None
                    elif len(centroid)==counter:
                        zeros=0
                        stable +=1
                        if stable==stable_frames:
                            # print(f"{counter} contours found")
                            return centroid
                    else:
                        counter=len(centroid)
                        stable=1
                        zeros=0
    def color_on_top(self, color):
        for i in range(10):
            ret,frame=self.cap.read()
        if ret:
            mask=self.get_mask(color, frame)
            region = mask[1,210:420]
            if np.any(region==255):
                return True
            else:
                return False
                
                    
if __name__ == "__main__":
    cam = Camera()
    cam.live_feed()
    