from src.config import load_settings
from src.maze_client import MazeApiError, MazeClient


def main() -> None:
    settings = load_settings()

    client = MazeClient(
        base_url=settings.base_url,
        authorization_header=settings.authorization_header,
    )

    print("Registering player...")

    try:
        player = client.register_player(settings.player_name)
        print("Player registered successfully.")
        print(player)
    except MazeApiError as error:
        error_message = str(error)

        if "already" in error_message.lower() or "409" in error_message:
            print("Player already registered. Continuing...")
        else:
            raise

    print("\nCurrent player:")
    print(client.get_player())

    print("\nAvailable mazes:")
    mazes = client.list_mazes()

    for maze in mazes:
        print(maze)


if __name__ == "__main__":
    main()