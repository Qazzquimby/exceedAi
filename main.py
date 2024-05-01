# alphazero implementation

import torch

from connect2.connect2game import Connect2Game
from connect2.connect2model import Connect2Model
from trainer import Trainer, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    "batch_size": 64,
    "numIters": 500,  # Total number of training iterations
    "num_simulations": 100,  # Total number of MCTS simulations to run when deciding on a move to play
    "numEps": 100,  # Number of full games (episodes) to run during each iteration
    "numItersForTrainExamplesHistory": 20,
    "epochs": 2,  # Number of epochs of training per iteration
    "checkpoint_path": "latest.pth",  # location to save latest set of weights
}


def train():
    game = Connect2Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    model = Connect2Model(board_size, action_size, device)

    trainer = Trainer(game, model, args)
    trainer.learn()


def watch():
    game = Connect2Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    model = Connect2Model(board_size, action_size, device)
    model = load_checkpoint(model=model, folder=".", filename=args["checkpoint_path"])

    game.auto_play(player1=model, player2=model)


if __name__ == "__main__":
    # train()
    watch()
