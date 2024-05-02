import os
import numpy as np
from random import shuffle

import torch
import torch.optim as optim
from tqdm import tqdm

from core import device, checkpoints_dir
from monte_carlo_tree_search import MCTS


class Trainer:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)

        self.best_loss = float("inf")
        self.best_epoch = 0

    def execute_episode(self):
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()

        def generator():
            while True:
                yield

        for _ in tqdm(generator(), desc="Move", position=2, leave=False):
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

    def learn(self):
        for iteration in tqdm(
            range(1, self.args["numIters"] + 1),
            desc="Iteration",
            position=0,
            leave=False,
        ):
            train_examples = []

            for _episode in tqdm(
                range(self.args["numEps"]), desc="Episode", position=1, leave=False
            ):
                iteration_train_examples = self.execute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            current_loss = self.train(train_examples)

            if torch.isnan(current_loss):
                print("Loss is NaN. Rolling back.")
                self.load_checkpoint(
                    filename=self.args["checkpoint_path"], suffix="latest"
                )

            if current_loss < self.best_loss:
                diff = self.best_loss - current_loss
                self.best_loss = current_loss
                self.best_iter = iteration

                self.save_checkpoint(
                    filename=self.args["checkpoint_path"],
                    suffix=f"best_model_{iteration}",
                )
                print(
                    f"Iter {iteration}: New best model saved with loss {self.best_loss:.4f}. Improved by {diff:.4f}"
                )

            self.save_checkpoint(filename=self.args["checkpoint_path"], suffix="latest")

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        total_loss = None

        for _epoch in tqdm(
            range(self.args["epochs"]), desc="Epoch", position=2, leave=False
        ):
            self.model.train()

            num_batches = int(len(examples) / self.args["batch_size"])
            for _batch_idx in tqdm(
                range(num_batches), desc="Batch", position=3, leave=False
            ):
                sample_ids = np.random.randint(
                    len(examples), size=self.args["batch_size"]
                )
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).view(
                    self.args["batch_size"], self.model.size
                )
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                boards = boards.contiguous()  # .cuda()
                target_pis = target_pis.contiguous()  # .cuda()
                target_vs = target_vs.contiguous()  # .cuda()
                if device.type == "cuda":
                    boards = boards.cuda()
                    target_pis = target_pis.cuda()
                    target_vs = target_vs.cuda()

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
        return total_loss

    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        return loss

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
