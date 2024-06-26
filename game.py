import abc
from pprint import pprint

import numpy as np
from tqdm import tqdm

from monte_carlo_tree_search import mcts_select_action


class Game(abc.ABC):
    name = NotImplemented
    size = NotImplemented

    def __init__(self):
        pass

    def get_init_board(self):
        raise NotImplementedError

    def get_next_state(self, board, player, action):
        raise NotImplementedError

    def has_legal_moves(self, board):
        raise NotImplementedError

    def get_valid_moves(self, board):
        raise NotImplementedError

    def mask_invalid_moves(self, state, action_probs):
        valid_moves = self.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)
        return action_probs

    def is_win(self, board: np.ndarray, player: int):
        raise NotImplementedError

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
        raise NotImplementedError

    def auto_play(self, player1, player2, args, num_games=5, should_print=False):
        winners = []
        for game_num in tqdm(
            range(num_games), desc="Eval Game", position=1, leave=False
        ):
            player1_goes_first = game_num % 2 == 0
            winner = self.auto_play_game(
                player1,
                player2,
                args,
                should_print=should_print,
                player1_goes_first=player1_goes_first,
            )
            winners.append(winner)
        player_1_score = winners.count(1)
        player_2_score = winners.count(-1)
        frac_player_1_wins = winners.count(1) / (player_1_score + player_2_score)
        if should_print:
            print(f"Player 1 wins: {winners.count(1)}, {frac_player_1_wins*100}%")
            print(f"Player -1 wins: {winners.count(-1)}, {(1-frac_player_1_wins)*100}%")
        return frac_player_1_wins

    def auto_play_game(
        self,
        player1,
        player2,
        args,
        should_print=False,
        player1_goes_first: bool = True,
    ):
        if player1 is None or player2 is None:
            # playing with human, turning prints on
            should_print = True

        state = self.get_init_board()
        current_player = 1
        if not player1_goes_first:
            current_player = -1

        turn = 0
        while True:
            board_for_player = self.get_board_from_perspective(state, current_player)

            if current_player == 1:
                current_player_model = player1
            else:
                current_player_model = player2

            if current_player_model is None:
                action = int(input("Enter your move: "))
            else:
                action = mcts_select_action(
                    model=current_player_model,
                    board_for_player=board_for_player,
                    game=self,
                    args=args,
                    temperature=0.1,
                )

            state, next_player = self.get_next_state(state, current_player, action)
            if should_print:
                print(f"turn {turn}:")
                pprint(state)

            reward = self.get_reward_for_player(state, current_player)

            if reward is not None:
                winner = current_player * reward
                if should_print:
                    print(f"Player {winner} wins")
                return winner

            current_player = next_player
            turn += 1
