
from arm_control import QArmTicTacToe


if __name__ == "__main__":
    bot = QArmTicTacToe()

    bot.pick_place_hasan("H1", 'A2')

   #bot.pick_place_hasan("H2", 'B3')

    #bot.pick_place_hasan("H3", 'C2')
    
    #bot.pick_place_hasan("H4", 'C1')
    
    #bot.pick_place_hasan("H5", 'A2')

    bot.terminate()