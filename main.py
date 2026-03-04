import cv2
from camera import Camera

# Camera().live_feed()
# Camera().live_depth()
cam=Camera()
cam.choose_hsv_color("green")
# cam.live_RGB("green")
cam.draw_contours('green')