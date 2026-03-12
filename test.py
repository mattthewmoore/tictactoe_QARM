
from arm_control import QArmTicTacToe

# ==========================================================
# 1. MANUAL CALIBRATION DATA
# ==========================================================
# [X, Y, Z] coordinates in meters. 
# Move the arm to each cell manually to find these exact values.

                  # Safe height to travel over pieces

z_loc_h12 = 0.06
z_loc_h345 = z_loc_h12
z_loc_a = 0.07
z_loc_b = 0.055
z_loc_c = 0.1
x_loc_b = .41

LOCATIONS = {
     'A1': [0.285, -0.1618, z_loc_a], 'A2': [0.2843, -0.0158, z_loc_a], 'A3': [0.2843, 0.16, z_loc_a],
     'B1': [x_loc_b, -0.1618, z_loc_b], 'B2': [x_loc_b, -0.0158, z_loc_b], 'B3': [x_loc_b, 0.1393, z_loc_b],
     'C1': [0.53, -0.1618, z_loc_c], 'C2': [0.53, -0.0158, z_loc_c], 'C3': [0.5, 0.1393, z_loc_c],
     'HOME': [0.45, 0.0, 0.49], 'CAM_PHI_POS':[-0.05, -0.15, 0.5, 0.0], 
     'H1': [0.0, -0.2413, z_loc_h12], 'H2': [0.1270, -0.2413, z_loc_h12], 'H3': [0.0, -0.3810, z_loc_h345],
     'H4': [0.127, -0.381, z_loc_h345], 'H5': [0.2540, -0.3810, z_loc_h345],
     'C3_PHI': [ 0.2432,  0.6897,  0.0330,  -1.0], 'C2_PHI' : [-0.0100,  0.6989,  0.0207,  -1.0],
     'C1_PHI': [-0.3000,  0.7096,  0.0269,  1.0]
 }

bot.move_to_phi(LOCATIONS['CAM_PHI_POS'], grip_cmd=0, duration=3.0)

if __name__ == "__main__":
    bot = QArmTicTacToe(LOCATIONS)

    bot.pick_place_hasan("H1", 'A2')

    bot.pick_place_hasan("H2", 'B3')

    bot.pick_place_hasan("H3", 'C2')
    
    bot.pick_place_hasan("H4", 'C1')
    
    bot.pick_place_hasan("H5", 'B1')

    bot.terminate()