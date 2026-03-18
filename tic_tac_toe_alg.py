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

    def available_moves(self, board):
        '''
        Gets and returns an arary of available moves from a NumPy array with
        robot and human player moves.

        Parameters:
            board: A Numpy array that has a 0 if the space is empty
            a 1 if the player has a move in that space, and a 2 if the robot has a
            move in that space.

        Returns:
            available_moves: A list of available moves
        '''
        available_moves = []
        for i,val in enumerate(board):
            if val==0:
                available_moves.append(i)
        return available_moves
    
    def winner(self, board):
        ''' 
        Returns a 2 if the robot wins, a 1 if the user wins, and none if no one has one.

        Parameters:
            board: A Numpy array that has a 0 if the space is empty
            a 1 if the player has a move in that space, and a 2 if the robot has a
            move in that space.
        
        Returns:
            board[a]: returns the value at index 'a' which will result in a 2 or 1
        
        '''
        for a, b, c in self.WIN_LINES:
            if board[a] != self.EMPTY and board[a] == board[b] == board[c]:
                return board[a]
        return None

    def is_draw(self, board):
        '''
        Returns True or False based on if there is a draw or not

        Parameters:
            board: A Numpy array that has a 0 if the space is empty
            a 1 if the player has a move in that space, and a 2 if the robot has a
            move in that space.
        
        Returns:
            True or False
        '''
        return self.winner(board) is None and all(c != self.EMPTY for c in board)

    def apply_move(self, board, idx, player):
        '''
        Mutates the board, Returns True if the move was applied, and False if it was
        an illegal move.

        Parameters:
            idx: The index of the move to be applied
            board: A Numpy array that has a 0 if the space is empty
            a 1 if the player has a move in that space, and a 2 if the robot has a
            move in that space.
            player: the number value of either a 1 for the human or 2 for the robot

        Returns:
            True or False depending if the move was applied or not
        '''
        if 0 <= idx < 9 and board[idx] == self.EMPTY:
            board[idx] = player
            return True
        return False

    def minimax(self, board, is_robot_turn, depth=0):
        '''
        Returns a score from the Robot's perspective:
            ROBOT win: +10 - depth
            HUMAN win: depth - 10
            draw: 0
        
        Parameters:
            board: A Numpy array that has a 0 if the space is empty
            a 1 if the player has a move in that space, and a 2 if the robot has a
            move in that space.
            is_robot_turn: Boolean, True if it is the robots turn and vise versa
            depth: the amount of turns for a given move sequence

        Returns:
            best: The best garunteed score assuming both players play perfect
        '''
        
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
        '''
        Returns which cell has the best robot move.

        Parameters:
            moves: A Numpy array that has a 0 if the space is empty
            a 1 if the player has a move in that space, and a 2 if the robot has a
            move in that space.

        Returns:
            best_move: The cell which has the best robot move.
        '''
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

    