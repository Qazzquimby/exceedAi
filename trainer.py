import numpy as np

import torch
from lightning import seed_everything
from torch.utils.data import DataLoader, random_split
import lightning as L

from core import checkpoints_dir
from monte_carlo_tree_search import MCTS


class TrainLoopManager:

    def __init__(self, game, model, args):
        # alternates between performing self-play games
        # retraining neural net on game trajectories
        # and evaluating new model against previous model.

        seed_everything(1)

        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)

        self.trainer = L.Trainer(
            max_epochs=self.args["epochs"],
            log_every_n_steps=1,
        )

    def run_train_loop(self, start_iter=1):
        train_examples = []
        for play_session in range(start_iter + 1, self.args["numIters"] + 1):
            train_examples += self.run_self_play()

            avg_moves_per_game = sum(
                [len(history) for history in train_examples]
            ) / len(train_examples)
            print(f"Avg Moves: {avg_moves_per_game:.3f}")

            self.fit_on_histories(train_examples)

    def fit_on_histories(self, train_examples):
        # rename
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.array([i[0] for i in train_examples])),
            torch.FloatTensor(np.array([i[1] for i in train_examples])),
            torch.FloatTensor(np.array([i[2] for i in train_examples])),
        )
        validation_frac = 0.1
        train_set, validation_set = random_split(
            dataset,
            [
                len(dataset) - int(len(dataset) * validation_frac),
                int(len(dataset) * validation_frac),
            ],
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=1,
        )
        validation_loader = DataLoader(
            validation_set,
            num_workers=1,
        )
        new_model = self.model.__class__()
        new_model.load_state_dict(self.model.state_dict())

        # todo consider training multiple models and taking the one with lowest val/loss
        self.trainer.fit(new_model, train_loader, validation_loader)

        frac_wins = self.game.auto_play(
            player1=new_model, player2=self.model, args=self.args, num_games=20
        )

        if frac_wins > 0.5:
            print(f"Accepting new model: {frac_wins}")
            self.save_checkpoint(
                filename=self.args["checkpoint_path"],
                suffix=f"best_model",
            )
            self.model = new_model
        else:
            print(f"Rejecting new model: {frac_wins}")

    def run_self_play(self):
        train_examples = []
        self.model.eval()
        for _game in range(self.args["numEps"]):
            #     tqdm(
            #     range(self.args["numEps"]), desc="Game", position=1, leave=False
            # ):
            iteration_train_examples = self.run_training_game()
            train_examples.extend(iteration_train_examples)
        return train_examples

    def run_training_game(self):
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()

        def generator():
            while True:
                yield

        while True:
            # for _ in tqdm(generator(), desc="Move", position=2, leave=False):
            board_from_current_player_perspective = (
                self.game.get_board_from_perspective(state, current_player)
            )

            self.mcts = MCTS(self.game, self.model, self.args)
            root = self.mcts.run(
                self.model, board_from_current_player_perspective, to_play=1
            )

            action_probs = [0 for _ in range(self.game.get_action_size())]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count
                # you use highest visit count, not value, because a node visited few times could have a very inaccurate value.

            action_probs /= np.sum(action_probs)
            train_examples.append(
                (board_from_current_player_perspective, current_player, action_probs)
            )

            action = root.select_action(temperature=0)
            state, current_player = self.game.get_next_state(
                state, current_player, action
            )
            reward = self.game.get_reward_for_player(state, current_player)

            if reward is not None:  # todo want to reward using expected value
                ret = []
                for (
                    hist_state,
                    hist_current_player,
                    hist_action_probs,
                ) in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append(
                        (
                            hist_state,
                            hist_action_probs,
                            reward * ((-1) ** (hist_current_player != current_player)),
                        )
                    )
                    # TODO the hell is this reward signal? Could just be multiplied by 1 or -1?

                return ret

    def save_checkpoint(self, filename, suffix=None):
        if suffix is not None:
            filename = f"{filename}_{suffix}"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        filepath = f"{checkpoints_dir / filename}.pth"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, filename, suffix=None):
        self.model = load_checkpoint(model=self.model, filename=filename, suffix=suffix)


def load_checkpoint(model, filename, suffix=None):
    if suffix is not None:
        filename = f"{filename}_{suffix}"
    filepath = f"{checkpoints_dir / filename}.pth"
    print(f"LOADING {filepath}")
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
