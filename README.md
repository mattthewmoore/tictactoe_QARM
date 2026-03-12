Physical Hardware
       [ROBOT BASE]
         (0,0,0)
            |
    ---------------------
    |  A1  |  A2  |  A3  |  Row A
    ---------------------
    |  B1  |  B2  |  B3  |  Row B
    ---------------------
    |  C1  |  C2  |  C3  |  Row C
    ---------------------
      Col 1  Col 2  Col 3



Structure of Project is as follows:

1.)Human starts first, When a new piece is detected wait 5 seconds to make a move.

2.)Robot takes photo input of board runs through ChatGPT/LLM and Determines next move

3.)Robot Picks and Places Figurines 

4.)Photo of board, check if game is over, if not make a move repeating steps, 2-3

5.) Change LED of Base when game is over



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

