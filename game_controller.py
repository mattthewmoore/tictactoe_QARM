from camera import Camera
from arm_control import QArmTicTacToe
from tic_tac_toe_alg import TicTacToe
import numpy as np
import sys

class Game_Controller():
   
    def __init__(self):
        self.z_loc_h12 = 0.06
        self.z_loc_h345 = self.z_loc_h12
        self.z_loc_a = 0.07
        self.z_loc_b = 0.055
        self.z_loc_c = 0.1
        self.x_loc_b = .41


        self.LOCATIONS = {
            'A1': [0.285, -0.1618, self.z_loc_a], 'A2': [0.2843, -0.0158, self.z_loc_a], 'A3': [0.2843, 0.16, self.z_loc_a],
            'B1': [self.x_loc_b, -0.1618, self.z_loc_b], 'B2': [self.x_loc_b, -0.0158, self.z_loc_b], 'B3': [self.x_loc_b, 0.1393, self.z_loc_b],
            'C1': [0.53, -0.1618, self.z_loc_c], 'C2': [0.53, -0.0158, self.z_loc_c], 'C3': [0.5, 0.1393, self.z_loc_c],
            'HOME': [0.45, 0.0, 0.49], 'CAM_PHI_POS':[-0.05, -0.15, 1.175, 0.0], 
            'H1': [0.0, -0.2413, self.z_loc_h12], 'H2': [0.1270, -0.2413, self.z_loc_h12], 'H3': [0.0, -0.3810, self.z_loc_h345],
            'H4': [0.127, -0.381, self.z_loc_h345], 'H5': [0.2540, -0.3810, self.z_loc_h345],
            'C3_PHI': [ 0.2432,  0.6897,  0.0330,  -1.0], 'C2_PHI' : [-0.0100,  0.6989,  0.0207,  -1.0],
            'C1_PHI': [-0.3000,  0.7096,  0.0269,  1.0]
        }

        self.cam = Camera()
        self.TTT = TicTacToe()
        self.centroids=np.zeros((9,2))
        self.moves=np.zeros(9)  #zeros for empty, 1 for human, 2 for robot
        self.bot = QArmTicTacToe(self.LOCATIONS)
        self.bot.move_to_phi(self.LOCATIONS['CAM_PHI_POS'], grip_cmd=0, duration=3.0)
        self.board_start = self.set_centroids("green")
        self.robot_piece_count = 1
        self.last_robot_move = 0
        self.last_human_move = 0
        self.game_start = 0
  
    def run_game(self):
        
        while True:

            winner = self.TTT.winner(self.moves)
            if winner == 1:
                print("Human Wins")
                try:
                    while True:
                        val=input("Please clear the board, then press enter to continue playing, to quit enter q")
                        if val == "":
                                self.moves.fill(0)
                                self.robot_piece_count = 1
                                self.last_robot_move = 0
                                self.last_human_move = 0
                                self.game_start = 0
                                break
                        elif val =="q":

                            raise KeyboardInterrupt
                        else:
                            continue
                except KeyboardInterrupt:
                    print("Game Over, Exiting...")
                    self.bot.terminate()
                    break
                continue
                
            if winner == 2:
                print("Robot Wins")
                self.bot.victory_dance()
                try:
                    while True:
                        val=input("Please clear the board, then press enter to continue playing, to quit enter q")
                        if val == "":
                                self.moves.fill(0)
                                self.robot_piece_count = 1
                                self.last_robot_move = 0
                                self.last_human_move = 0
                                self.game_start = 0
                                break
                        elif val =="q":

                            raise KeyboardInterrupt
                        else:
                            continue
                except KeyboardInterrupt:
                    print("Game Over, Exiting...")
                    self.bot.terminate()
                    break
                continue
                
            if self.TTT.is_draw(self.moves):
                print("Draw")
                try:
                    while True:
                        val=input("Please clear the board, then press enter to continue playing, to quit enter q")
                        if val == "":
                                self.moves.fill(0)
                                self.robot_piece_count = 1
                                self.last_robot_move = 0
                                self.last_human_move = 0
                                self.game_start = 0
                                break
                        elif val =="q":

                            raise KeyboardInterrupt
                        else:
                            continue
                except KeyboardInterrupt:
                    print("Game Over, Exiting...")
                    self.bot.terminate()
                    break
                continue
                        
            if self.game_start == 0:
                if input ("Press '1' for the player to start first and '2' for the robot to start first") == '2':
                    self.find_missing()
                    winner = self.TTT.winner(self.moves)
                    self.game_start +=1      
                else:
                    input("Press Enter when you finish placing your move") == ""
                    winner = self.TTT.winner(self.moves)
                    self.find_missing()
                    self.game_start +=1
                    if winner == 1:
                        try:
                            while True:
                                val=input("Please clear the board, then press enter to continue playing, to quit enter q")
                                if val == "":
                                        self.moves.fill(0)
                                        self.robot_piece_count = 1
                                        self.last_robot_move = 0
                                        self.last_human_move = 0
                                        self.game_start = 0
                                        break
                                elif val =="q":

                                    raise KeyboardInterrupt
                                else:
                                    continue
                        except KeyboardInterrupt:
                            print("Game Over, Exiting...")
                            self.bot.terminate()
                            break
                        continue
                    if self.TTT.is_draw(self.moves):
                        print("Draw")
                        try:
                            while True:
                                val=input("Please clear the board, then press enter to continue playing, to quit enter q")
                                if val == "":
                                        self.moves.fill(0)
                                        self.robot_piece_count = 1
                                        self.last_robot_move = 0
                                        self.last_human_move = 0
                                        self.game_start = 0
                                        break
                                elif val =="q":

                                    raise KeyboardInterrupt
                                else:
                                    continue
                        except KeyboardInterrupt:
                            print("Game Over, Exiting...")
                            self.bot.terminate()
                            break
                        continue
            else:
                    if input("Press Enter when you finish placing your move") == "":
                        winner = self.TTT.winner(self.moves)
                        self.find_missing()
                        self.game_start +=1
                    if winner == 1:
                        print("Human Wins")
                        try:
                            while True:
                                val=input("Please clear the board, then press enter to continue playing, to quit enter q")
                                if val == "":
                                        self.moves.fill(0)
                                        self.robot_piece_count = 1
                                        self.last_robot_move = 0
                                        self.last_human_move = 0
                                        self.game_start = 0
                                        break
                                elif val =="q":

                                    raise KeyboardInterrupt
                                else:
                                    continue
                        except KeyboardInterrupt:
                            print("Game Over, Exiting...")
                            self.bot.terminate()
                            break
                        continue
                    if self.TTT.is_draw(self.moves):
                        print("Draw")
                        try:
                            while True:
                                val=input("Please clear the board, then press enter to continue playing, to quit enter q")
                                if val == "":
                                        self.moves.fill(0)
                                        self.robot_piece_count = 1
                                        self.last_robot_move = 0
                                        self.last_human_move = 0
                                        self.game_start = 0
                                        break
                                elif val =="q":

                                    raise KeyboardInterrupt
                                else:
                                    continue
                        except KeyboardInterrupt:
                            print("Game Over, Exiting...")
                            self.bot.terminate()
                            break
                        continue

            self.last_robot_move = self.get_computer_move()
            self.move_robot()
            self.robot_piece_count+=1
        sys.exit()        
    # def record_player_move(self):
    #     current_board = self.set_centroids()
    #     for i in self.board_start:
    #         if 

    def get_computer_move(self):
        move = self.TTT.best_robot_move(self.moves)
        # print(f"computer move{move}")
        self.last_robot_move = move
        self.moves[move] = 2
        return move
    
    def set_centroids(self, color):
        centroids=self.cam.stable_frame_centroids(color)
        centroids=self.cam.order_centroid(centroids)
        # print(centroids)
        return centroids
    
    def find_missing(self):
        original = self.board_start
        new = self.set_centroids("green")

        for i, (cx, cy) in enumerate(original):
            # calc distance from this original to all new centroids
            distances = np.sqrt((new[:, 0] - cx)**2 + (new[:, 1] - cy)**2)
            
            if np.min(distances) > 20:  # no close match found
                if self.moves[i] !=2:
                    self.moves[i] = 1
                    self.last_human_move = i+1
            if self.moves[4] == 1:
                if not self.cam.color_on_top("blue") and not self.cam.color_on_top("gold"):
                    self.moves[1] = 0
                    self.last_human_move = 0
        # print(self.moves)
    
    def move_robot(self):
        key = ''
        if self.last_robot_move == 0:
            key = 'C3'
        elif self.last_robot_move == 1:
            key = 'C2'
        elif self.last_robot_move == 2:
            key = 'C1'
        elif self.last_robot_move == 3:
            key = 'B3'
        elif self.last_robot_move == 4:
            key = 'B2'
        elif self.last_robot_move == 5:
            key = 'B1'
        elif self.last_robot_move == 6:
            key = 'A3'
        elif self.last_robot_move == 7:
            key = 'A2'
        elif self.last_robot_move == 8:
            key = 'A1'
        self.bot.pick_place_hasan(f"H{self.robot_piece_count}", key)
        self.bot.move_to_phi(self.LOCATIONS['CAM_PHI_POS'], grip_cmd=0, duration=3.0)
        # self.robot_piece_count +=1

        #self.bot.pick_place_hasan("H1", 'C3')
    
    def moves_list(self):
        print(self.moves)