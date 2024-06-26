from connect2.connect2game import Connect2Game
from connect2.connect2model import Connect2Model
from connect4.connect4game import Connect4Game
from connect4.connect4model import Connect4Model
from core import checkpoints_dir, DEBUG
from trainer import TrainLoopManager, load_checkpoint

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

    model = Connect2Model()
    args["checkpoint_path"] = get_with_run_name("connect2", run_name)
    return game, model


def get_connect_4_game_model_path(run_name=None):
    game = Connect4Game()
    model = Connect4Model()
    args["checkpoint_path"] = get_with_run_name("connect4", run_name)
    return game, model


def get_best_checkpoint_suffix():
    suffix = f"best_model"
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
            game=game,
            filename=get_best_checkpoint_suffix(),
        )
    except FileNotFoundError:
        print("No checkpoint found. Training from scratch.")
    trainer = TrainLoopManager(game, model, args)
    trainer.run_train_loop()  # start_iter=get_start_iter())


def watch(game_name: str, run_name=None):
    game, model = get_game_model(game_name, run_name)

    model = load_checkpoint(
        model=model,
        game=game,
        filename=get_best_checkpoint_suffix(),
    )

    game.auto_play(player1=model, player2=model, args=args)


def player_vs_model(game_name, run_name=None):
    game, model = get_game_model(game_name, run_name)

    model = load_checkpoint(
        model=model,
        game=game,
        filename=get_best_checkpoint_suffix(),
    )

    game.auto_play(player1=model, player2=None, args=args)


def main():
    game_name = "connect4"
    run_name = ""
    if DEBUG:
        run_name += "_debug"
    train(game_name=game_name, run_name=run_name)
    watch(game_name=game_name, run_name=run_name)
    player_vs_model(game_name=game_name, run_name=run_name)


if __name__ == "__main__":
    main()
