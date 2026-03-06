from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities
import time

# ==========================================================
# 1. MANUAL CALIBRATION DATA
# ==========================================================
# [X, Y, Z] coordinates in meters. 
# Move the arm to each cell manually to find these exact values.
LOCATIONS = {
     'A1': [0.2843, -0.1618, 0.04], 'A2': [0.2843, -0.0158, 0.04], 'A3': [0.2843, 0.1393, 0.04],
     'B1': [0.4143, -0.1618, 0.025], 'B2': [0.4143, -0.0158, 0.025], 'B3': [0.4143, 0.1393, 0.025],
     'C1': [0.53, -0.1618, 0.05], 'C2': [0.53, -0.0158, 0.05], 'C3': [0.53, 0.1393, 0.05],
     'HOME': [0.45, 0.0, 0.49], 'CAM_PHI_POS':[-0.05, -0.15, 1.175, 0.0]
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
        print("QArm Initialized and Ready.")

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
