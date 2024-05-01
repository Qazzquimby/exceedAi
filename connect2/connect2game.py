import numpy as np

from connect2.connect2model import Connect2Model


# Todo, also need an object for the state. It can't just be a 4array


class Connect2Game:
    """
    A very, very simple game of ConnectX in which we have:
        rows: 1
        columns: 4
        winNumber: 2
    """

    def __init__(self):
        self.columns = 4
        self.win = 2

    def get_init_board(self):
        b = np.zeros((self.columns,), dtype=int)
        return b

    def get_board_size(self):
        return self.columns

    def get_action_size(self):
        return self.columns

    def get_next_state(self, board, player, action):
        b = np.copy(board)
        if b[action] == 0:
            b[action] = player

        # new board, new player
        return (b, -player)

    def has_legal_moves(self, board):
        for index in range(self.columns):
            if board[index] == 0:
                return True
        return False

    def get_valid_moves(self, board):
        # All moves are invalid by default
        valid_moves = [0] * self.get_action_size()

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

    def get_reward_for_player(self, board, player):
        # return None if not ended, 1 if player the given player 1, -1 if they lost

        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if self.has_legal_moves(board):
            return None

        return 0

    def get_board_from_perspective(self, board, player):
        return player * board

    def auto_play(self, player1: Connect2Model, player2: Connect2Model):
        num_games = 5

        for game_num in range(num_games):
            self.auto_play_game(player1, player2)

    def auto_play_game(self, player1: Connect2Model, player2: Connect2Model):
        state = self.get_init_board()
        current_player = 1

        turn = 0
        while True:
            board_for_player = self.get_board_from_perspective(state, current_player)

            if current_player == 1:
                action = player1.select_action(board_for_player)
            else:
                action = player2.select_action(board_for_player)

            state, next_player = self.get_next_state(state, current_player, action)
            print(f"turn {turn}: {state}")

            reward = self.get_reward_for_player(state, current_player)

            if reward is not None:
                if reward == 1:
                    print(f"Player {current_player} wins")
                elif reward == -1:
                    print(f"Player {current_player*-1} wins")
                else:
                    print("Draw")
                return

            current_player = next_player
            turn += 1
