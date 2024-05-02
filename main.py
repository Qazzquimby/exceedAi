# alphazero implementation

from connect2.connect2game import Connect2Game
from connect2.connect2model import Connect2Model
from connect4.connect4game import Connect4Game
from connect4.connect4model import Connect4Model
from core import device
from trainer import Trainer, load_checkpoint

args = {
    "batch_size": 16,  # 64,
    "numIters": 500,  # Total number of training iterations
    "num_simulations": 20,  # Total number of MCTS simulations to run when deciding on a move to play
    "numEps": 20,  # Number of full games (episodes) to run during each iteration
    "epochs": 2,  # Number of epochs of training per iteration
    # "checkpoint_path": "latest.pth",  # location to save latest set of weights
}


def get_connect_2_game_model_path():
    game = Connect2Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    model = Connect2Model(board_size, action_size, device)
    args["checkpoint_path"] = "connect2.pth"
    return game, model


def get_connect_4_game_model_path():
    game = Connect4Game()
    board_size = (game.rows, game.columns)
    action_size = game.get_action_size()

    model = Connect4Model(board_size, action_size, device)
    args["checkpoint_path"] = "connect4.pth"
    return game, model


def train():
    # game, model = get_connect_2_game_model_path()
    game, model = get_connect_4_game_model_path()
    try:
        model = load_checkpoint(
            model=model, folder=".", filename=args["checkpoint_path"]
        )
    except FileNotFoundError:
        pass
    trainer = Trainer(game, model, args)
    trainer.learn()


def watch():
    # game, model = get_connect_2_game_model_path()
    game, model = get_connect_4_game_model_path()

    model = load_checkpoint(model=model, folder=".", filename=args["checkpoint_path"])

    game.auto_play(player1=model, player2=model)


if __name__ == "__main__":
    train()
    watch()
