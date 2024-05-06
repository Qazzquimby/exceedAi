# alphazero implementation

from connect2.connect2game import Connect2Game
from connect2.connect2model import Connect2Model
from connect4.connect4game import Connect4Game
from connect4.connect4model import Connect4Model
from core import checkpoints_dir
from trainer import Trainer, load_checkpoint

batch_size = 16

args = {
    "batch_size": 64,  # 64,
    "numIters": 2500,  # Total number of training iterations
    "num_simulations": 400,  # Total number of MCTS simulations to run when deciding on a move to play
    "numEps": batch_size * 2,
    # Number of full games (episodes) to run during each iteration
    "epochs": 2,  # Number of epochs of training per iteration
    # "checkpoint_path": "latest.pth",  # location to save latest set of weights
}


def get_with_run_name(base_name: str, run_name=None):
    if run_name:
        return f"{base_name}_{run_name}"
    else:
        return base_name


def get_connect_2_game_model_path(run_name):
    game = Connect2Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    model = Connect2Model(board_size, action_size)
    args["checkpoint_path"] = get_with_run_name("connect2", run_name)
    return game, model


def get_connect_4_game_model_path(run_name):
    game = Connect4Game()
    board_size = (game.rows, game.columns)
    action_size = game.get_action_size()

    model = Connect4Model(board_size, action_size)
    args["checkpoint_path"] = get_with_run_name("connect4", run_name)
    return game, model


def get_best_checkpoint_suffix():
    last_iter = get_start_iter() - 1
    suffix = f"best_model_{last_iter}"
    return suffix


def get_start_iter():
    checkpoint_paths = list(checkpoints_dir.glob("*.pth"))
    best_checkpoints_for_game = [
        path
        for path in checkpoint_paths
        if args["checkpoint_path"] in path.name and "best_model" in path.name
    ]
    iters = [
        int(path.name.split("_")[-1].split(".")[0])
        for path in best_checkpoints_for_game
    ]
    if iters:
        return max(iters) + 1
    return 1


def get_game_model(game_name: str, run_name=None):
    if game_name == "connect2":
        game, model = get_connect_2_game_model_path(run_name=run_name)
    elif game_name == "connect4":
        game, model = get_connect_4_game_model_path(run_name=run_name)
    else:
        raise ValueError(f"Unknown game name: {game_name}")
    return game, model


def train(game_name: str, run_name=None):
    game, model = get_game_model(game_name, run_name)
    try:
        model = load_checkpoint(
            model=model,
            filename=args["checkpoint_path"],
            suffix=get_best_checkpoint_suffix(),
        )
    except FileNotFoundError:
        print("No checkpoint found. Training from scratch.")
    trainer = Trainer(game, model, args)
    trainer.learn(start_iter=get_start_iter())


def watch(game_name: str, run_name=None):
    game, model = get_game_model(game_name, run_name)

    model = load_checkpoint(
        model=model,
        filename=args["checkpoint_path"],
        suffix=get_best_checkpoint_suffix(),
    )

    game.auto_play(player1=model, player2=model, args=args)


def player_vs_model(game_name, run_name=None):
    game, model = get_game_model(game_name, run_name)

    model = load_checkpoint(
        model=model,
        filename=args["checkpoint_path"],
        suffix="latest",  # get_best_checkpoint_suffix(),
    )

    game.auto_play(player1=model, player2=None, args=args)


if __name__ == "__main__":
    game_name = "connect2"
    run_name = "lightning"
    train(game_name=game_name, run_name=run_name)
    watch(game_name=game_name, run_name=run_name)
    player_vs_model(game_name=game_name, run_name=run_name)
