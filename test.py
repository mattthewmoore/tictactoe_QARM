from arm_control import QArmTicTacToe
import numpy as np
import time
import camera



LOCATIONS = {
     'A1': [0.2843, -0.1618, 0.04], 'A2': [0.2843, -0.0158, 0.04], 'A3': [0.2843, 0.1393, 0.04],
     'B1': [0.4143, -0.1618, 0.025], 'B2': [0.4143, -0.0158, 0.025], 'B3': [0.4143, 0.1393, 0.025],
     'C1': [0.53, -0.1618, 0.05], 'C2': [0.53, -0.0158, 0.05], 'C3': [0.53, 0.1393, 0.05],
     'HOME': [0.45, 0.0, 0.49], 'CAM_PHI_POS':[-0.05, -0.15, 1.175, 0.0]
 }


bot = QArmTicTacToe()

cam = camera.Camera()


bot.move_to_phi(LOCATIONS['CAM_PHI_POS'], grip_cmd=0, duration=2.0)
time.sleep(1)

bot.myArm.terminate()

cam.live_feed()






#bot.move_to_xyz(LOCATIONS['HOME'], grip_cmd=0, duration=2.0)
#time.sleep(1)
#bot.move_to_xyz(LOCATIONS['CAM'], grip_cmd=0, duration=2.0)
#time.sleep(1)
#
# bot.set_gripper(grip_cmd=1, duration=0.5)
# time.sleep(1)
# bot.move_to_xyz(LOCATIONS['HOME'], grip_cmd=1, duration=2.0)
# time.sleep(1)
# bot.move_to_xyz(LOCATIONS['A1'], grip_cmd=1, duration=2.0)
# time.sleep(1)
# bot.set_gripper(grip_cmd=0, duration=0.5)
# time.sleep(1)
# bot.myArm.terminate()



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