from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities
from quanser.hardware import MAX_STRING_LENGTH
import numpy as np
import time


# ==========================================================
#  ROBOT CONTROL CLASS
# ==========================================================
class QArmTicTacToe:
    """Control-only interface for commanding the QArm from other files."""

    def __init__(self,Location_Dict):
        self.myArm = QArm(hardware=1)
        self.myUtils = QArmUtilities()
        self.sampleTime = 1/200
        self._saved_gripper_cmd = 0.5
        self.LOCATIONS = Location_Dict
        print("QArm Initialized and Ready.")
    ## Victory Dance ##
    def victory_dance(self):
        """Big exaggerated victory dance at the camera pose."""
        open_grip = 0.0
        closed_grip = 0.7

        cam = list(self.LOCATIONS['CAM_PHI_POS'])

        # Start at camera pose
        self.move_to_phi(cam, grip_cmd=open_grip, duration=1.0)

        # Big exaggerated nodding:
        # nod = mainly change phi2 and phi3 together
        # phi3 is the "1.175" joint entry from CAM_PHI_POS
        nod_down = [cam[0], -0.55, 0.72, 0.0]
        nod_up   = [cam[0],  0.05, 1.25, 0.0]

        # Do several big nods, not just one
        for _ in range(3):
            self.move_to_phi(nod_down, grip_cmd=open_grip, duration=0.45)
            self.move_to_phi(nod_up,   grip_cmd=open_grip, duration=0.45)

        # Add a side-to-side celebration after nodding
        left_pose  = [-0.55, -0.10, 1.05, 0.0]
        right_pose = [ 0.45, -0.10, 1.05, 0.0]

        for _ in range(2):
            self.move_to_phi(left_pose,  grip_cmd=open_grip, duration=0.40)
            self.move_to_phi(right_pose, grip_cmd=open_grip, duration=0.40)

        # Gripper celebration
        for _ in range(3):
            self.set_gripper(closed_grip, duration=0.30)
            self.set_gripper(open_grip,   duration=0.30)

        # Return to camera pose
            self.move_to_phi(cam, grip_cmd=open_grip, duration=1.0)

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
            # LED Blue (0,0,1) while moving
            self.myArm.read_write_std(phiCMD=phi_cmd, grpCMD=grip_cmd, baseLED=[0, 0, 1])
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
            # LED Blue (0,0,1) while moving in joint space
            self.myArm.read_write_std(phiCMD=phi_cmd, grpCMD=grip_cmd, baseLED=[0, 0, 1])
            # Maintain 200Hz loop
            time.sleep(self.sampleTime - (time.time() - loop_start) % self.sampleTime)

    def move_to_cell(self, cell_key, grip_cmd, duration=2.0):
        """Moves to a named board location from self.LOCATIONS."""
        if cell_key not in self.LOCATIONS:
            raise KeyError(f"Unknown cell key: {cell_key}")
        self.move_to_xyz(self.LOCATIONS[cell_key], grip_cmd, duration)

    def read_phi(self):
        """Returns the current measured joint angles [phi1, phi2, phi3, phi4]."""
        return list(self.myArm.measJointPosition[0:4])

    def read_xyz(self):
        """Returns the current measured end-effector position [x, y, z] in meters."""
        phi_now = self.read_phi()
        xyz, _ = self.myUtils.qarm_forward_kinematics(phi_now)
        return list(xyz)

    def home(self, grip_cmd=0, duration=2.0):
        """Moves to the HOME point defined in self.LOCATIONS."""
        self.move_to_cell('HOME', grip_cmd, duration)

    def set_gripper(self, grip_cmd, duration=0.5):
        """Holds the current pose while commanding only gripper state."""
        start_time = time.time()
        phi_cmd = self.myArm.measJointPosition[0:4]
        while time.time() - start_time < duration:
            loop_start = time.time()
            time.sleep(self.sampleTime - (time.time() - loop_start) % self.sampleTime)

    def terminate(self):
        """Releases QArm hardware resources."""
        # LED green (0,0,1) while doness
        self.myArm.terminate()


    def pick_place_hasan(self, h_key, position):
        "Pickup hassan pieces from staging area and place in position"

        grip_cmd_head = 0.7 # Closed gripper command

        if h_key not in self.LOCATIONS:
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
            self.move_to_xyz(self.LOCATIONS[h_key], grip_cmd=0, duration=3.0)

            #Close gripper to pick up piece
            self.set_gripper(grip_cmd=grip_cmd_head, duration=0.5)

            #Adjust lift to avoid collisions with H5
            self.move_to_phi([-2.0,self.read_phi()[1],0.5,0.0], grip_cmd=grip_cmd_head, duration=3.0)

        elif h_key == 'H2' or h_key == 'H5':
            #Turn slightly sideways to pick up piece
            self.move_to_phi([-1.1,0.0,0.5,0.0], grip_cmd=0, duration=3.0)

            #Move to staging area above piece
            self.move_to_xyz(self.LOCATIONS[h_key], grip_cmd=0, duration=3.0)

            #Close gripper to pick up piece
            self.set_gripper(grip_cmd=grip_cmd_head, duration=0.5)

        elif h_key == 'H4':
            #Turn slightly sideways to pick up piece
            self.move_to_phi([-1.0,0.0,0.5,0.0], grip_cmd=0, duration=3.0)

            #Move to staging area above piece
            self.move_to_xyz(self.LOCATIONS[h_key], grip_cmd=0, duration=3.0)

            #Close gripper to pick up piece
            self.set_gripper(grip_cmd=grip_cmd_head, duration=0.5)

            #Adjust lift to avoid collisions with H5
            self.move_to_phi([-1.45,0.0,0.5,0.0], grip_cmd=grip_cmd_head, duration=3.0)

        #Return to home position with piece
        self.move_to_xyz(self.LOCATIONS['HOME'], grip_cmd=grip_cmd_head, duration=3.0)

    def place_H(self, position, grip_cmd_head=0.7):
        "Place H pieces in position with adjustments as needed for height and place in position"

        if 'A' in position:
            #adjust place for A row
            self.move_to_phi([0,-0.5,1.2,0.0], grip_cmd=grip_cmd_head, duration=3.0)

            #Move to target position with piece
            self.move_to_xyz(self.LOCATIONS[position], grip_cmd=grip_cmd_head, duration=3.0)

            #Drop piece at target position
            self.set_gripper(grip_cmd=0, duration=0.5)

        elif 'B' in position:

            #Move to target position with piece
            self.move_to_xyz(self.LOCATIONS[position], grip_cmd=grip_cmd_head, duration=3.0)

            #Drop piece at target position
            self.set_gripper(grip_cmd=0, duration=0.5)

        elif 'C' in position:

            pos = position[1:2] #Extract cell number 

            phi = 'C' + pos + '_PHI' #Get corresponding PHI key for C row

            #adjust place for C row with PHI adjustments
            self.move_to_phi(self.LOCATIONS[phi], grip_cmd=grip_cmd_head, duration=3.0)

            #Move to target position with piece
            self.move_to_xyz(self.LOCATIONS[position], grip_cmd=grip_cmd_head, duration=3.0)

            self.set_gripper(grip_cmd=0, duration=0.5)


        
        #rotate up to avoid collisions when moving back to home
        self.move_to_phi([self.read_phi()[0],0.0,0.0,0.0], grip_cmd=0, duration=2.0)

