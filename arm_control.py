from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities
from quanser.hardware import MAX_STRING_LENGTH
import numpy as np
import time

# ==========================================================
# 1. MANUAL CALIBRATION DATA
# ==========================================================
# [X, Y, Z] coordinates in meters. 
# Move the arm to each cell manually to find these exact values.
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

STAGING_AREA = [0.100, -0.150, 0.025] # Where the robot's pieces are kept
Z_HOVER = 0.120                       # Safe height to travel over pieces

# ==========================================================
# 2. ROBOT CONTROL CLASS
# ==========================================================
class QArmTicTacToe:
    """Control-only interface for commanding the QArm from other files."""

    def __init__(self):
        self.myArm = QArm(hardware=1)
        self.myUtils = QArmUtilities()
        self.sampleTime = 1/200
        self._saved_gripper_cmd = 0.5
        print("QArm Initialized and Ready.")

    def _set_mode(self, mode):
        """Set all joints to Position mode (0) or PWM mode (1)."""
        mode = int(mode)
        if mode not in (0, 1):
            raise ValueError("mode must be 0 (position) or 1 (pwm)")

        mode_arr = np.array([mode, mode, mode, mode, mode], dtype=np.uint8)
        mode_options = (
            f"j0_mode={mode};j1_mode={mode};j2_mode={mode};j3_mode={mode};gripper_mode={mode};"
            "j0_profile_config=0;j0_profile_velocity=1.5708;j0_profile_acceleration=1.0472;"
            "j1_profile_config=0;j1_profile_velocity=1.5708;j1_profile_acceleration=1.0472;"
            "j2_profile_config=0;j2_profile_velocity=1.5708;j2_profile_acceleration=1.0472;"
            "j3_profile_config=0;j3_profile_velocity=1.5708;j3_profile_acceleration=1.0472;"
        )

        self.myArm.mode = mode_arr
        self.myArm.card.set_card_specific_options(mode_options, MAX_STRING_LENGTH)

    def enable_teach_mode(self):
        """Relax joints for hand-guiding and continue updating measured states."""
        self.myArm.read_std()
        self._saved_gripper_cmd = float(np.clip(self.myArm.measJointPosition[4], 0.1, 0.9))
        self._set_mode(1)

        # In PWM mode, commanding zeros keeps joints backdrivable.
        self.myArm.read_write_std(
            phiCMD=self.myArm.measJointPosition[0:4],
            grpCMD=self._saved_gripper_cmd,
            baseLED=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        )

    def disable_teach_mode(self):
        """Return to position mode and hold current pose where the arm was taught."""
        self.myArm.read_std()
        phi_hold = list(self.myArm.measJointPosition[0:4])
        grip_hold = float(np.clip(self.myArm.measJointPosition[4], 0.1, 0.9))
        self._set_mode(0)
        self.move_to_phi(phi_hold, grip_cmd=grip_hold, duration=0.25)

    def snapshot_phi(self):
        """Read and return current measured joint angles [phi1, phi2, phi3, phi4]."""
        self.myArm.read_std()
        return list(self.myArm.measJointPosition[0:4])

    def move_to_xyz(self, target_xyz, grip_cmd, duration=2.0):
        """Moves the arm to a Task Space coordinate smoothly over a set duration."""
        start_time = time.time()
        # Solve IK using current joint state as a seed for stability
        _, phi_cmd = self.myUtils.qarm_inverse_kinematics(target_xyz, 0, self.myArm.measJointPosition[0:4])
        
        while time.time() - start_time < duration:
            loop_start = time.time()
            # LED Green (0,1,0) while moving
            self.myArm.read_write_std(phiCMD=phi_cmd, grpCMD=grip_cmd, baseLED=[0, 1, 0])
            # Maintain 200Hz loop
            time.sleep(self.sampleTime - (time.time() - loop_start) % self.sampleTime)

    def move_to_phi(self, target_phi, grip_cmd, duration=2.0):
        """Moves the arm to a Joint Space target (phi) over a set duration."""
        if len(target_phi) != 4:
            raise ValueError("target_phi must contain 4 joint values")

        start_time = time.time()
        phi_cmd = list(target_phi)

        while time.time() - start_time < duration:
            loop_start = time.time()
            # LED Cyan (0,1,1) while moving in joint space
            self.myArm.read_write_std(phiCMD=phi_cmd, grpCMD=grip_cmd, baseLED=[0, 1, 1])
            # Maintain 200Hz loop
            time.sleep(self.sampleTime - (time.time() - loop_start) % self.sampleTime)

    def move_to_cell(self, cell_key, grip_cmd, duration=2.0):
        """Moves to a named board location from LOCATIONS."""
        if cell_key not in LOCATIONS:
            raise KeyError(f"Unknown cell key: {cell_key}")
        self.move_to_xyz(LOCATIONS[cell_key], grip_cmd, duration)

    def read_phi(self):
        """Returns the current measured joint angles [phi1, phi2, phi3, phi4]."""
        return list(self.myArm.measJointPosition[0:4])

    def read_xyz(self):
        """Returns the current measured end-effector position [x, y, z] in meters."""
        phi_now = self.read_phi()
        xyz, _ = self.myUtils.qarm_forward_kinematics(phi_now)
        return list(xyz)

    def home(self, grip_cmd=0, duration=2.0):
        """Moves to the HOME point defined in LOCATIONS."""
        self.move_to_cell('HOME', grip_cmd, duration)

    def set_gripper(self, grip_cmd, duration=0.5):
        """Holds the current pose while commanding only gripper state."""
        start_time = time.time()
        phi_cmd = self.myArm.measJointPosition[0:4]
        while time.time() - start_time < duration:
            loop_start = time.time()
            self.myArm.read_write_std(phiCMD=phi_cmd, grpCMD=grip_cmd, baseLED=[0, 0, 1])
            time.sleep(self.sampleTime - (time.time() - loop_start) % self.sampleTime)

    def terminate(self):
        """Releases QArm hardware resources."""
        self.myArm.terminate()


    def pick_place_hasan(self, h_key, position):
        "Pickup hassan pieces from staging area and place in position"

        grip_cmd_head = 0.7 # Closed gripper command

        if h_key not in LOCATIONS:
            raise KeyError(f"Unknown cell key: {h_key}")    
        
        #Pickup piece and go to home position
        self.pickup_H(h_key, grip_cmd_head=grip_cmd_head)

        #Move to target position with piece an place piece down
        self.place_H(position, grip_cmd_head=grip_cmd_head)

    def pickup_H(self,h_key, grip_cmd_head=0.7):
        "Pickup H pieces from staging area with adjustments as needed for height and place in position"

        if h_key == 'H1' or h_key == 'H3':
            #Turn to face sideways to pick up piece
            self.move_to_phi([-1.45,0.0,1.25,0.0], grip_cmd=0, duration=3.0)

            #Move to staging area above piece
            self.move_to_xyz(LOCATIONS[h_key], grip_cmd=0, duration=3.0)

            #Close gripper to pick up piece
            self.set_gripper(grip_cmd=grip_cmd_head, duration=0.5)

            #Adjust lift to avoid collisions with H5
            self.move_to_phi([-2.0,self.read_phi()[1],0.5,0.0], grip_cmd=grip_cmd_head, duration=3.0)

        elif h_key == 'H2' or h_key == 'H5':
            #Turn slightly sideways to pick up piece
            self.move_to_phi([-1.1,0.0,0.5,0.0], grip_cmd=0, duration=3.0)

            #Move to staging area above piece
            self.move_to_xyz(LOCATIONS[h_key], grip_cmd=0, duration=3.0)

            #Close gripper to pick up piece
            self.set_gripper(grip_cmd=grip_cmd_head, duration=0.5)

        elif h_key == 'H4':
            #Turn slightly sideways to pick up piece
            self.move_to_phi([-1.0,0.0,0.5,0.0], grip_cmd=0, duration=3.0)

            #Move to staging area above piece
            self.move_to_xyz(LOCATIONS[h_key], grip_cmd=0, duration=3.0)

            #Close gripper to pick up piece
            self.set_gripper(grip_cmd=grip_cmd_head, duration=0.5)

            #Adjust lift to avoid collisions with H5
            self.move_to_phi([-1.45,0.0,0.5,0.0], grip_cmd=grip_cmd_head, duration=3.0)

        #Return to home position with piece
        self.move_to_xyz(LOCATIONS['HOME'], grip_cmd=grip_cmd_head, duration=3.0)

    def place_H(self, position, grip_cmd_head=0.7):
        "Place H pieces in position with adjustments as needed for height and place in position"

        if 'A' in position:
            #adjust place for A row
            self.move_to_phi([0,-0.5,1.2,0.0], grip_cmd=grip_cmd_head, duration=3.0)

            #Move to target position with piece
            self.move_to_xyz(LOCATIONS[position], grip_cmd=grip_cmd_head, duration=3.0)

            #Drop piece at target position
            self.set_gripper(grip_cmd=0, duration=0.5)

        elif 'B' in position:

            #Move to target position with piece
            self.move_to_xyz(LOCATIONS[position], grip_cmd=grip_cmd_head, duration=3.0)

            #Drop piece at target position
            self.set_gripper(grip_cmd=0, duration=0.5)

        elif 'C' in position:

            pos = position[1:2] #Extract cell number 

            phi = 'C' + pos + '_PHI' #Get corresponding PHI key for C row

            #adjust place for C row with PHI adjustments
            self.move_to_phi(LOCATIONS[phi], grip_cmd=grip_cmd_head, duration=3.0)

            #Move to target position with piece
            self.move_to_xyz(LOCATIONS[position], grip_cmd=grip_cmd_head, duration=3.0)

            self.set_gripper(grip_cmd=0, duration=0.5)


        
        #rotate up to avoid collisions when moving back to home
        self.move_to_phi([self.read_phi()[0],0.0,0.0,0.0], grip_cmd=0, duration=2.0)

