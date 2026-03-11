from arm_control import QArmTicTacToe
import numpy as np
import time
import camera

z_loc_h = 0.04


LOCATIONS = {
     'A1': [0.2843, -0.1618, 0.04], 'A2': [0.2843, -0.0158, 0.04], 'A3': [0.2843, 0.1393, 0.04],
     'B1': [0.4143, -0.1618, 0.025], 'B2': [0.4143, -0.0158, 0.025], 'B3': [0.4143, 0.1393, 0.025],
     'C1': [0.53, -0.1618, 0.05], 'C2': [0.53, -0.0158, 0.05], 'C3': [0.53, 0.1393, 0.05],
     'HOME': [0.45, 0.0, 0.49], 'CAM_PHI_POS':[-0.05, -0.15, 1.175, 0.0], 
     'H1': [0.0, -0.2413, z_loc_h], 'H2': [0.1270, -0.2413, z_loc_h], 'H3': [0.0, -0.3810, 0.025],
     'H4': [0.127, -0.381, 0.025], 'H5': [0.2540, -0.3810, 0.025] 
 }


bot = QArmTicTacToe()

#cam = camera.Camera()
bot.pick_place_hassan('H1', 'B2')

bot.pick_place_hassan('H2', 'A2')

bot.pick_place_hassan('H3', 'C2')

bot.pick_place_hassan('H4', 'A3')

bot.pick_place_hassan('H5', 'C3')

bot.myArm.terminate()

#cam.draw_contours('blue')




def get_stable_dept_data(self,centroid_arr, depth_frame):
    valid_readings = []
    depth = 0.0
    range = 5
    cx = centroid_arr(0)
    cy = centroid_arr(1)

    for i in range (-range,range+1):
        for j in range(-range, range+1):
            x = cx + i
            y = cy + j
            
            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                d = depth_frame[x,y]

                if d > 0:
                    valid_readings.append(d)
    average = float(np.mean(valid_readings))
    if valid_readings:
        depth = average
    return depth