from main import get_connect_4_game_model_path, args, get_best_checkpoint_suffix
from trainer import load_checkpoint


def latest_vs_best():
    game, latest_model = get_connect_4_game_model_path()
    latest_model = load_checkpoint(
        model=latest_model,
        game=game,
        filename="latest",
    )

    game, best_model = get_connect_4_game_model_path()
    best_model = load_checkpoint(
        model=best_model,
        game=game,
        filename=get_best_checkpoint_suffix(),
    )

    game.auto_play(player1=latest_model, player2=best_model, args=args, num_games=500)


if __name__ == "__main__":
    latest_vs_best()
