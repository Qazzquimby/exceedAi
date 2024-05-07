import numpy as np

from core import Game


class Connect2Game(Game):
    """
    A very, very simple game of ConnectX in which we have:
        rows: 1
        columns: 4
        winNumber: 2
    """

    name = "connect2"

    columns = 4
    win = 2

    size = columns
    action_size = columns

    def __init__(self):
        super().__init__()
        self.columns = 4
        self.win = 2

    def get_init_board(self):
        b = np.zeros((self.columns,), dtype=int)
        return b

    def get_next_state(self, board, player, action):
        b = np.copy(board)
        if b[action] == 0:
            b[action] = player

        # new board, new player
        return b, -player

    def has_legal_moves(self, board):
        for index in range(self.columns):
            if board[index] == 0:
                return True
        return False

    def get_valid_moves(self, board):
        # All moves are invalid by default
        valid_moves = [0] * self.action_size

        for index in range(self.columns):
            if board[index] == 0:
                valid_moves[index] = 1

        return valid_moves

    def is_win(self, board, player):
        count = 0
        for index in range(self.columns):
            if board[index] == player:
                count = count + 1
            else:
                count = 0

            if count == self.win:
                return True

        return False

    def get_board_from_perspective(self, board, player):
        return player * board
