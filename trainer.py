import pickle

import numpy as np

import torch
from lightning import seed_everything
from torch.utils.data import DataLoader, random_split
import lightning as L
from tqdm import tqdm

from core import checkpoints_dir, DEBUG
from game import Game
from monte_carlo_tree_search import MCTS

MAX_TRAIN_EXAMPLES = 5_000


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
        # todo right now mcts is always overwritten

        self.trainer = L.Trainer(
            max_epochs=self.args["epochs"],
            log_every_n_steps=1,
            enable_model_summary=False,
        )

        if DEBUG:
            self.trainer = L.Trainer(
                max_epochs=2,
                log_every_n_steps=1,
                fast_dev_run=True,
                detect_anomaly=True,
            )

    def run_train_loop(self, start_iter=1):
        train_examples = load_train_examples(self.game)

        if DEBUG:
            max_iterations = 2
        else:
            max_iterations = self.args["numIters"]

        for play_session in range(start_iter + 1, start_iter + max_iterations + 1):
            train_examples += self.run_self_play()
            train_examples = train_examples[-MAX_TRAIN_EXAMPLES:]
            save_train_examples(self.game, train_examples)

            self.fit_and_evaluate(train_examples)

    def fit_and_evaluate(self, train_examples):
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.array([i[0] for i in train_examples])),
            torch.FloatTensor(np.array([i[1] for i in train_examples])),
            torch.FloatTensor(np.array([i[2] for i in train_examples])),
        )

        best_new_model = self.fit_multiple(dataset=dataset)

        self_play_evaluation = False

        if self_play_evaluation:
            frac_wins = self.game.auto_play(
                player1=best_new_model, player2=self.model, args=self.args, num_games=20
            )

            if frac_wins >= 0.5:
                print(f"Accepting new model: {frac_wins}")
                save_checkpoint(
                    model=self.model,
                    game=self.game,
                    filename=f"best_model",
                )
                self.model = best_new_model
            else:
                print(f"Rejecting new model: {frac_wins}")
        else:
            if best_new_model.val_loss < self.model.val_loss:
                print(f"Accepting new model: {best_new_model.val_loss}")
                save_checkpoint(
                    model=self.model,
                    game=self.game,
                    filename=f"best_model",
                )
                self.model = best_new_model

    def fit_multiple(self, dataset, attempts=1):
        best_new_model = None
        for _ in range(attempts):
            new_model = self.fit_single(dataset)
            print(f"Training attempt {_}/{attempts} - val/loss: {new_model.val_loss}")
            if not best_new_model or new_model.val_loss < best_new_model.val_loss:
                best_new_model = new_model
        return best_new_model

    def fit_single(self, dataset):
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

        new_model.train()
        # todo consider training multiple models and taking the one with lowest val/loss
        self.trainer.fit(new_model, train_loader, validation_loader)
        return new_model

    def run_self_play(self):
        train_examples = []
        self.model.eval()

        if DEBUG:
            num_games = 2
        else:
            num_games = self.args["numEps"]
        with torch.no_grad():
            for _game in tqdm(range(num_games), desc="Game", position=1, leave=False):
                iteration_train_examples = self.run_training_game()
                train_examples.extend(iteration_train_examples)
        return train_examples

    def run_training_game(self):
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()

        while True:
            board_from_current_player_perspective = (
                self.game.get_board_from_perspective(state, current_player)
            )

            self.mcts = MCTS(self.game, self.model, self.args)
            root = self.mcts.run(
                self.model, board_from_current_player_perspective, to_play=1
            )

            action_probs = [0 for _ in range(self.game.action_size)]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count
                # you use the highest visit count, not value,
                # because a node visited few times could have a very inaccurate value.

            action_probs /= np.sum(action_probs)
            train_examples.append(
                (board_from_current_player_perspective, current_player, action_probs)
            )

            action = root.select_action(temperature=0)
            state, current_player = self.game.get_next_state(
                state, current_player, action
            )
            reward = self.game.get_reward_for_player(state, current_player)

            if reward is not None:
                ret = []
                for (
                    hist_state,
                    hist_current_player,
                    hist_action_probs,
                ) in train_examples:

                    old_version = reward * (
                        (-1) ** (hist_current_player != current_player)
                    )
                    # todo remove

                    if hist_current_player != current_player:
                        reward *= -1

                    assert old_version == reward

                    ret.append(
                        (
                            hist_state,
                            hist_action_probs,
                            reward * ((-1) ** (hist_current_player != current_player)),
                        )
                    )
                    # TODO could just be multiplied by 1 or -1?

                return ret


def save_checkpoint(model, game: Game, filename: str):
    filename += ".pth"
    path = checkpoints_dir / game.name / filename
    torch.save(
        {
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(model, game: Game, filename: str):
    filename += ".pth"
    path = checkpoints_dir / game.name / filename
    print(f"LOADING {path}")
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def get_train_examples_path(game):
    return checkpoints_dir / game.name / "train_examples.pkl"


def load_train_examples(game):
    try:
        train_examples = pickle.load(open(get_train_examples_path(game), "rb"))
    except FileNotFoundError:
        train_examples = []
    return train_examples


def save_train_examples(game, train_examples):
    path = get_train_examples_path(game)
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(train_examples, open(get_train_examples_path(game), "wb"))
