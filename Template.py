from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities
import time
import numpy as np
import cv2

# ==========================================================
# 1. MANUAL CALIBRATION DATA
# ==========================================================
# [X, Y, Z] coordinates in meters. 
# Move the arm to each cell manually to find these exact values.
LOCATIONS = {
    'A1': [0.120, -0.070, 0.025], 'A2': [0.120, 0.000, 0.025], 'A3': [0.120, 0.070, 0.025],
    'B1': [0.170, -0.070, 0.025], 'B2': [0.170, 0.000, 0.025], 'B3': [0.170, 0.070, 0.025],
    'C1': [0.220, -0.070, 0.025], 'C2': [0.220, 0.000, 0.025], 'C3': [0.220, 0.070, 0.025]
}

STAGING_AREA = [0.100, -0.150, 0.025] # Where the robot's pieces are kept
Z_HOVER = 0.120                       # Safe height to travel over pieces

# ==========================================================
# 2. ROBOT CONTROL CLASS
# ==========================================================
class QArmTicTacToe:
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

    def execute_pick_and_place(self, cell_key):
        """Full sequence: Pick from staging -> Hover -> Lower to Cell -> Release."""
        print(f"Robot moving to place piece at {cell_key}...")
        
        # --- PICK PHASE ---
        self.move_to_xyz([STAGING_AREA[0], STAGING_AREA[1], Z_HOVER], 0) # Hover over pieces
        self.move_to_xyz(STAGING_AREA, 0)                               # Lower
        self.move_to_xyz(STAGING_AREA, 1)                               # Grip (1 is closed)
        self.move_to_xyz([STAGING_AREA[0], STAGING_AREA[1], Z_HOVER], 1) # Lift
        
        # --- PLACE PHASE ---
        target_xyz = LOCATIONS[cell_key]
        self.move_to_xyz([target_xyz[0], target_xyz[1], Z_HOVER], 1)    # Hover over target
        self.move_to_xyz(target_xyz, 1)                                  # Lower
        self.move_to_xyz(target_xyz, 0)                                  # Release (0 is open)
        self.move_to_xyz([target_xyz[0], target_xyz[1], Z_HOVER], 0)    # Return to Safe Height

    def capture_board(self):
        """Captures image from the QArm camera."""
        # Use cv2 to grab the frame - camera index might vary (0, 1, or 2)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('current_board.jpg', frame)
            print("Board photo saved.")
        cap.release()
        return 'current_board.jpg'

    def get_llm_decision(self, image_path):
        """
        PLACEHOLDER for LLM Logic. 
        In your real implementation, you will send the image_path to ChatGPT/Vision API.
        """
        print("Analyzing board state with LLM...")
        time.sleep(2) # Simulating API latency
        
        # Logic would return (best_move, is_game_over)
        # For example: return 'B2', False
        return 'B2', False 

    def end_game_signal(self):
        """Sets Base LED to Magenta to signal the end of the game."""
        # Run for a short burst to ensure the hardware receives the command
        for _ in range(100):
            self.myArm.read_write_std(phiCMD=self.myArm.measJointPosition[0:4], 
                                     grpCMD=0, baseLED=[1, 0, 1])

# ==========================================================
# 3. MAIN GAME LOOP
# ==========================================================
def main():
    bot = QArmTicTacToe()
    
    try:
        while True:
            # Step 1: Human Move Wait
            print("\n>>> HUMAN TURN: Place your piece on the board.")
            input("Press [Enter] after you have made your move...")
            
            print("Detecting change... waiting 5 seconds for stability.")
            time.sleep(5)

            # Step 2: Perception
            image = bot.capture_board()
            
            # Step 3: Reasoning
            next_move, game_over = bot.get_llm_decision(image)
            
            if game_over:
                print("Game over detected by LLM!")
                bot.end_game_signal()
                break

            # Step 4: Action
            if next_move in LOCATIONS:
                bot.execute_pick_and_place(next_move)
            
            # Step 5: Final check after Robot move
            image = bot.capture_board()
            _, game_over = bot.get_llm_decision(image)
            
            if game_over:
                print("Robot move ended the game! Victory/Draw.")
                bot.end_game_signal()
                break

    except KeyboardInterrupt:
        print("\nInterrupt detected. Cleaning up...")
    finally:
        bot.myArm.terminate()

if __name__ == "__main__":
    main()