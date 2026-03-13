import cv2
from camera import Camera
from game_controller import Game_Controller

# Camera().live_feed()
# Camera().live_depth()
# cam=Camera()
# cam.color_on_top("blue")
# centroid= cam.draw_contours("gold")
# cam.choose_hsv_color("gold")
# print(centroid)
game = Game_Controller()
game.run_game()
# while True:
#     if input("Press Enter to continue...") == "":
#         # game.move_robot()
#         game.find_missing()
#         game.get_computer_move()
#         game.move_robot()
#         game.moves_list()
# cam.choose_hsv_color("blue")
# cam.live_RGB("green")q
# cam.draw_contours('blue')
# cam.get_depth_snapshot()