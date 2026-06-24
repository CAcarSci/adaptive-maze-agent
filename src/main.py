from src.bots.baseline_bot import BaselineMazeBot
from src.bots.smart_bot import SmartMazeBot
from src.config import Settings, load_settings
from src.data.telemetry_logger import TelemetryLogger
from src.maze_client import MazeApiError, MazeClient


def ensure_clean_player(
    *,
    client: MazeClient,
    player_name: str,
    reset_player_on_start: bool,
) -> None:
    """
    During development the player can remain inside a maze after a failed or
    interrupted run.

    If reset_player_on_start is enabled, the player is reset at the start of
    each run. This makes local comparison between baseline and smart bot easier.
    """
    if reset_player_on_start:
        print("RESET_PLAYER_ON_START=true. Resetting player state...")
        try:
            client.forget_player()
        except MazeApiError:
            pass

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


def create_bot(
    *,
    settings: Settings,
    client: MazeClient,
    telemetry_logger: TelemetryLogger,
) -> BaselineMazeBot:
    if settings.bot_type == "baseline":
        return BaselineMazeBot(
            client=client,
            telemetry_logger=telemetry_logger,
        )

    if settings.bot_type == "smart":
        return SmartMazeBot(
            client=client,
            telemetry_logger=telemetry_logger,
        )

    raise ValueError(
        f"Unsupported BOT_TYPE='{settings.bot_type}'. "
        "Use 'baseline' or 'smart'."
    )


def main() -> None:
    settings = load_settings()

    client = MazeClient(
        base_url=settings.base_url,
        authorization_header=settings.authorization_header,
    )

    ensure_clean_player(
        client=client,
        player_name=settings.player_name,
        reset_player_on_start=settings.reset_player_on_start,
    )

    maze_name = settings.default_maze_name
    print(f"Selected maze: {maze_name}")
    print(f"Selected bot type: {settings.bot_type}")

    telemetry_logger = TelemetryLogger(
        output_path="experiments/action_logs.csv"
    )

    bot = create_bot(
        settings=settings,
        client=client,
        telemetry_logger=telemetry_logger,
    )

    try:
        bot.solve(maze_name)
    except MazeApiError as error:
        if "already played this maze" in str(error).lower():
            print(
                f"Maze '{maze_name}' was already played with this token. "
                "Choose another maze in DEFAULT_MAZE_NAME or enable "
                "RESET_PLAYER_ON_START=true for local development."
            )
            return

        raise

    print("\nCurrent player after run:")
    print(client.get_player())

    print("\nTelemetry written to:")
    print("experiments/action_logs.csv")


if __name__ == "__main__":
    main()