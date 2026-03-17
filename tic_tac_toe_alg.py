# Code was modified from AI generated algorithm

class TicTacToe:
    def __init__(self):
        self.EMPTY = 0
        self.HUMAN = 1
        self.ROBOT = 2

        self.WIN_LINES = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
            (0, 4, 8), (2, 4, 6)               # diagonals
        ]

    # def new_board(sel,board):
    #     return [self.EMPTY] * 9

    # def print_board(self, board):
    #     def cell(i):
    #         return board[i] if board[i] != self.EMPTY else str(i)
    #     print()
    #     print(f" {cell(0)} | {cell(1)} | {cell(2)} ")
    #     print("---+---+---")
    #     print(f" {cell(3)} | {cell(4)} | {cell(5)} ")
    #     print("---+---+---")
    #     print(f" {cell(6)} | {cell(7)} | {cell(8)} ")
    #     print()

    def available_moves(self, board):
        available_moves = []
        for i,val in enumerate(board):
            if val==0:
                available_moves.append(i)
        return available_moves
    
    def winner(self, board):
        for a, b, c in self.WIN_LINES:
            if board[a] != self.EMPTY and board[a] == board[b] == board[c]:
                return board[a]
        return None

    def is_draw(self, board):
        return self.winner(board) is None and all(c != self.EMPTY for c in board)

    def apply_move(self, board, idx, player):
        """Mutates board. Returns True if move applied, False if illegal."""
        if 0 <= idx < 9 and board[idx] == self.EMPTY:
            board[idx] = player
            return True
        return False

    def minimax(self, board, is_robot_turn, depth=0):
        """
        Returns a score from ROBOT's perspective:
          ROBOT win: +10 - depth
          HUMAN win: depth - 10
          draw: 0
        """
        w = self.winner(board)
        if w == self.ROBOT:
            return 10 - depth
        if w == self.HUMAN:
            return depth - 10
        if not self.available_moves(board):
            return 0

        if is_robot_turn:
            best = -999
            for m in self.available_moves(board):
                board[m] = self.ROBOT
                score = self.minimax(board, False, depth + 1)
                board[m] = self.EMPTY
                best = max(best, score)
            return best
        else:
            best = 999
            for m in self.available_moves(board):
                board[m] = self.HUMAN
                score = self.minimax(board, True, depth + 1)
                board[m] = self.EMPTY
                best = min(best, score)
            return best

    def best_robot_move(self, moves):
        best_score = -999
        best_move = 20
        board = list(moves)

        for m in self.available_moves(board):
            board[m] = self.ROBOT
            score = self.minimax(board, False, depth=1)
            board[m] = self.EMPTY

            if score > best_score:
                best_score = score
                best_move = m
        moves[best_move] = 2
        return best_move

    # def get_human_move(self, board):
    #     while True:
    #         try:
    #             idx = int(input("Your move (0-8): ").strip())
    #             if idx in self.available_moves(board):
    #                 return idx
    #             print("That square is not available.")
    #         except ValueError:
    #             print("Please enter a number 0-8.")

    # def game_loop(self):
    #     board = self.new_board()
    #     turn = self.HUMAN  # human starts; change to self.ROBOT if you want robot first

    #     while True:
    #         self.print_board(board)

    #         w = self.winner(board)
    #         if w is not None:
    #             print(f"{w} wins!")
    #             break
    #         if self.is_draw(board):
    #             print("Draw!")
    #             break

    #         if turn == self.HUMAN:
    #             move = self.get_human_move(board)
    #             self.apply_move(board, move, self.HUMAN)
    #             turn = self.ROBOT
    #         else:
    #             move = self.best_robot_move(board)
    #             print(f"Robot plays: {move}")
    #             self.apply_move(board, move, self.ROBOT)
    #             turn = self.HUMAN
