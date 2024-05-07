import numpy as np

from game import Game


class Connect4Game(Game):
    name = "connect4"

    rows = 6
    columns = 7
    size = rows * columns
    action_size = columns
    win = 4

    def __init__(self):
        super().__init__()

    def get_init_board(self):
        board = np.zeros((self.rows, self.columns), dtype=int)
        return board

    def get_next_state(self, board, player, action):
        new_board = np.copy(board)

        for row in range(self.rows - 1, -1, -1):
            if new_board[row][action] == 0:
                new_board[row][action] = player
                return new_board, -player

        assert False, "Invalid move"

    def has_legal_moves(self, board: np.ndarray):
        for column in range(self.columns):
            if board[0][column] == 0:
                return True
        return False

    def get_valid_moves(self, board):
        # All moves are invalid by default
        valid_moves = [0] * self.action_size

        for column in range(self.columns):
            if board[0][column] == 0:
                valid_moves[column] = 1

        return valid_moves

    def is_win(self, board: np.ndarray, player: int):
        rows = len(board)
        columns = len(board[0])

        # Check horizontal
        for r in range(rows):
            for c in range(columns - 3):
                if all(board[r][c + i] == player for i in range(4)):
                    return True

        # Check vertical
        for c in range(columns):
            for r in range(rows - 3):
                if all(board[r + i][c] == player for i in range(4)):
                    return True

        # Check diagonal (top-left to bottom-right)
        for r in range(rows - 3):
            for c in range(columns - 3):
                if all(board[r + i][c + i] == player for i in range(4)):
                    return True

        # Check diagonal (bottom-left to top-right)
        for r in range(3, rows):
            for c in range(columns - 3):
                if all(board[r - i][c + i] == player for i in range(4)):
                    return True

        return False

    def get_board_from_perspective(self, board, player):
        return player * board
