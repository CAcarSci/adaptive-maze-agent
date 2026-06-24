from src.bots.baseline_bot import BaselineMazeBot
from src.config import load_settings
from src.maze_client import MazeApiError, MazeClient


def ensure_clean_player(client: MazeClient, player_name: str) -> None:
    """
    During development the player can remain inside a maze after a failed or
    interrupted run. For Step 1 we reset the player to make runs reproducible.
    """
    try:
        player = client.get_player()

        if player.get("isInMaze"):
            print(
                f"Player is currently in maze '{player.get('maze')}'. "
                "Resetting player state..."
            )
            client.forget_player()

    except MazeApiError:
        # Player may not exist yet. That is fine.
        pass

    try:
        client.register_player(player_name)
        print("Player registered successfully.")
    except MazeApiError as error:
        error_message = str(error)

        if "already" in error_message.lower() or "409" in error_message:
            print("Player already registered. Continuing...")
        else:
            raise


def main() -> None:
    settings = load_settings()

    client = MazeClient(
        base_url=settings.base_url,
        authorization_header=settings.authorization_header,
    )

    ensure_clean_player(client, settings.player_name)

    maze_name = settings.default_maze_name
    print(f"Selected maze: {maze_name}")

    bot = BaselineMazeBot(client=client)

    try:
        bot.solve(maze_name)
    except MazeApiError as error:
        if "already played this maze" in str(error).lower():
            print(
                f"Maze '{maze_name}' was already played with this token. "
                "Choose another maze in DEFAULT_MAZE_NAME."
            )
            return

        raise

    print("\nCurrent player after run:")
    print(client.get_player())


if __name__ == "__main__":
    main()