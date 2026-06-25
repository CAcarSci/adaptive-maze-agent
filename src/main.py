import argparse
import os
from pathlib import Path
from typing import Any, Sequence

from src.bots.baseline_bot import BaselineMazeBot
from src.bots.decision_tree_bot import DecisionTreeMazeBot
from src.bots.smart_bot import SmartMazeBot
from src.config import Settings, load_settings
from src.data.telemetry_logger import TelemetryLogger
from src.maze_client import MazeApiError, MazeClient


DEFAULT_TELEMETRY_OUTPUT_PATH = Path("experiments/action_logs.csv")

DEFAULT_TRAINING_MAZES = [
    "Example Maze",
    "Gradius Pathways",
    "Hello Maze",
]

TRAINING_BOT_TYPES = [
    "baseline",
    "smart",
]

INTERACTIVE_BOT_OPTIONS = [
    ("baseline", "baseline_dfs"),
    ("smart", "reward_aware"),
    ("decision_tree", "decision_tree"),
]


def print_guide() -> None:
    print(
        """
Adaptive Maze Agent

This application solves maze challenges using different bot policies.

Game objective:
  1. Register a player.
  2. Enter a maze.
  3. Move through the maze.
  4. Collect rewards.
  5. Find the exit.
  6. Exit with the collected score.

Implemented bot policies:
  - baseline_dfs:
      Deterministic DFS-style exploration baseline.

  - reward_aware:
      Explainable heuristic policy based on telemetry signals.

  - decision_tree:
      Lightweight ML policy trained from telemetry-derived labels.

Recommended workflow:
  1. Collect training telemetry from baseline and reward-aware bots.
  2. Analyze telemetry.
  3. Train the Decision Tree policy.
  4. Evaluate all policies on seen and unseen mazes.
  5. Review generated reports.

Interactive mode:
  Run the application without arguments:

  python -m src.main

  The interactive menu lets you:
    - see the current player
    - register or switch player
    - list available mazes
    - play a new maze with a selected bot
    - collect training telemetry
    - train the Decision Tree
    - evaluate all policies
    - run the full pipeline with default settings

Common CLI commands:
  python -m src.main list-mazes

  python -m src.main run-bot \\
    --bot-type baseline \\
    --maze-name "Example Maze" \\
    --reset-player

  python -m src.main collect-training-data --fresh-telemetry

  python -m src.main analyze-telemetry

  python -m src.main train-decision-tree

  python -m src.main evaluate

  python -m src.main pipeline --fresh-telemetry

Generated files:
  - experiments/action_logs.csv
  - reports/telemetry_analysis.md
  - models/decision_tree_policy.joblib
  - reports/decision_tree_policy.txt
  - reports/decision_tree_policy.png
  - reports/decision_tree_training_report.md
  - reports/evaluation_results.csv
  - reports/evaluation_report.md

Notes:
  - Decision Tree telemetry is excluded from training inside the training script.
  - Evaluation telemetry is written separately to experiments/evaluation_action_logs.csv.
  - Reports are generated deterministically from telemetry and metrics.
"""
    )


def print_welcome(player_name: str) -> None:
    print("\n" + "=" * 80)
    print("Welcome to Adaptive Maze Agent")
    print("=" * 80)
    print(f"Current known player name: {player_name}")
    print("")
    print("You can play individual mazes, collect telemetry, train the Decision Tree,")
    print("or run the full reporting pipeline from this menu.")
    print("=" * 80)


def create_client(settings: Settings) -> MazeClient:
    return MazeClient(
        base_url=settings.base_url,
        authorization_header=settings.authorization_header,
    )


def resolve_player_name(settings: Settings) -> str:
    """
    Resolve the active player name.

    Priority:
    1. PLAYER_NAME from environment / .env
    2. Interactive user input
    3. Settings fallback value

    The resolved value is stored in os.environ for the current process so
    downstream modules that call load_settings() use the same player name.
    """

    env_player_name = os.getenv("PLAYER_NAME")

    if env_player_name and env_player_name.strip():
        player_name = env_player_name.strip()
        os.environ["PLAYER_NAME"] = player_name
        return player_name

    print("\nNo PLAYER_NAME was found in the environment.")
    entered_name = input(
        f"Enter player name [{settings.player_name}]: "
    ).strip()

    player_name = entered_name or settings.player_name
    os.environ["PLAYER_NAME"] = player_name

    return player_name


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
    each run. This makes local comparison between bot policies easier.
    """

    if reset_player_on_start:
        print("Resetting player state...")
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
        pass

    try:
        client.register_player(player_name)
        print(f"Player registered successfully: {player_name}")
    except MazeApiError as error:
        error_message = str(error)

        if "already" in error_message.lower() or "409" in error_message:
            print(f"Player already registered: {player_name}")
        else:
            raise


def register_new_player_interactively(current_player_name: str) -> str:
    settings = load_settings()
    client = create_client(settings)

    print("\nCurrent known player name:")
    print(f"  {current_player_name}")

    should_register_new_player = ask_yes_no(
        "Do you want to register a new player?",
        default=False,
    )

    if not should_register_new_player:
        print("Keeping current player.")
        return current_player_name

    new_player_name = input("Enter new player name: ").strip()

    if not new_player_name:
        print("No player name entered. Keeping current player.")
        return current_player_name

    print(f"\nSwitching player to: {new_player_name}")
    print("Resetting player state for this token...")

    try:
        client.forget_player()
    except MazeApiError:
        pass

    try:
        client.register_player(new_player_name)
        print(f"Player registered successfully: {new_player_name}")
    except MazeApiError as error:
        error_message = str(error)

        if "already" in error_message.lower() or "409" in error_message:
            print(f"Player already registered: {new_player_name}")
        else:
            raise

    os.environ["PLAYER_NAME"] = new_player_name

    return new_player_name


def normalize_bot_type(bot_type: str) -> str:
    normalized = bot_type.strip().lower()

    aliases = {
        "baseline": "baseline",
        "baseline_dfs": "baseline",
        "dfs": "baseline",
        "smart": "smart",
        "reward_aware": "smart",
        "reward-aware": "smart",
        "decision_tree": "decision_tree",
        "decision-tree": "decision_tree",
        "tree": "decision_tree",
    }

    if normalized not in aliases:
        raise ValueError(
            f"Unsupported bot type '{bot_type}'. "
            "Use 'baseline', 'smart' or 'decision_tree'."
        )

    return aliases[normalized]


def create_bot(
    *,
    bot_type: str,
    client: MazeClient,
    telemetry_logger: TelemetryLogger,
) -> BaselineMazeBot:
    normalized_bot_type = normalize_bot_type(bot_type)

    if normalized_bot_type == "baseline":
        return BaselineMazeBot(
            client=client,
            telemetry_logger=telemetry_logger,
        )

    if normalized_bot_type == "smart":
        return SmartMazeBot(
            client=client,
            telemetry_logger=telemetry_logger,
        )

    if normalized_bot_type == "decision_tree":
        return DecisionTreeMazeBot(
            client=client,
            telemetry_logger=telemetry_logger,
        )

    raise ValueError(f"Unsupported bot type '{bot_type}'.")


def fetch_mazes() -> list[dict[str, Any]]:
    settings = load_settings()
    client = create_client(settings)

    return client.list_mazes()


def get_maze_display_name(maze: dict[str, Any]) -> str:
    return str(maze.get("name") or maze.get("mazeName") or "unknown")


def print_maze_table(mazes: list[dict[str, Any]]) -> None:
    print("\nAvailable mazes:")
    print("-" * 95)
    print(f"{'#':<4} {'Maze':<32} {'Tiles':<10} {'Potential reward':<18}")
    print("-" * 95)

    for index, maze in enumerate(mazes, start=1):
        maze_name = get_maze_display_name(maze)
        total_tiles = maze.get("totalTiles", "n/a")
        potential_reward = maze.get("potentialReward", "n/a")

        print(
            f"{index:<4} "
            f"{maze_name:<32} "
            f"{str(total_tiles):<10} "
            f"{str(potential_reward):<18}"
        )

    print("-" * 95)


def list_mazes() -> None:
    mazes = fetch_mazes()
    print_maze_table(mazes)


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    default_label = "Y/n" if default else "y/N"

    while True:
        answer = input(f"{prompt} [{default_label}]: ").strip().lower()

        if not answer:
            return default

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please enter yes or no.")


def ask_number(
    *,
    prompt: str,
    minimum: int,
    maximum: int,
) -> int:
    while True:
        value = input(prompt).strip()

        try:
            number = int(value)
        except ValueError:
            print(f"Please enter a number between {minimum} and {maximum}.")
            continue

        if minimum <= number <= maximum:
            return number

        print(f"Please enter a number between {minimum} and {maximum}.")


def select_maze_interactively() -> str:
    mazes = fetch_mazes()
    print_maze_table(mazes)

    selected_index = ask_number(
        prompt="Select maze number: ",
        minimum=1,
        maximum=len(mazes),
    )

    selected_maze = mazes[selected_index - 1]
    return get_maze_display_name(selected_maze)


def select_bot_type_interactively() -> str:
    print("\nAvailable bot policies:")
    print("-" * 70)

    for index, (bot_type, policy_name) in enumerate(INTERACTIVE_BOT_OPTIONS, start=1):
        print(f"{index}. {bot_type:<15} policy={policy_name}")

    print("-" * 70)

    selected_index = ask_number(
        prompt="Select bot number: ",
        minimum=1,
        maximum=len(INTERACTIVE_BOT_OPTIONS),
    )

    return INTERACTIVE_BOT_OPTIONS[selected_index - 1][0]


def select_multiple_mazes_interactively() -> list[str]:
    mazes = fetch_mazes()
    print_maze_table(mazes)

    print(
        "\nEnter maze numbers separated by commas, "
        "or press Enter to use the default training mazes."
    )
    print(f"Default training mazes: {', '.join(DEFAULT_TRAINING_MAZES)}")

    value = input("Maze numbers: ").strip()

    if not value:
        return DEFAULT_TRAINING_MAZES

    selected_mazes: list[str] = []

    for raw_part in value.split(","):
        raw_part = raw_part.strip()

        if not raw_part:
            continue

        try:
            index = int(raw_part)
        except ValueError:
            print(f"Ignoring invalid maze number: {raw_part}")
            continue

        if 1 <= index <= len(mazes):
            selected_mazes.append(get_maze_display_name(mazes[index - 1]))
        else:
            print(f"Ignoring out-of-range maze number: {raw_part}")

    if not selected_mazes:
        print("No valid maze numbers selected. Using default training mazes.")
        return DEFAULT_TRAINING_MAZES

    return selected_mazes


def run_bot_once(
    *,
    bot_type: str,
    maze_name: str,
    telemetry_output_path: Path,
    reset_player_on_start: bool,
    player_name: str | None = None,
) -> None:
    settings = load_settings()
    resolved_player_name = player_name or resolve_player_name(settings)
    client = create_client(settings)

    ensure_clean_player(
        client=client,
        player_name=resolved_player_name,
        reset_player_on_start=reset_player_on_start,
    )

    print("\nRun configuration:")
    print(f"  Player: {resolved_player_name}")
    print(f"  Maze: {maze_name}")
    print(f"  Bot type: {bot_type}")
    print(f"  Telemetry output: {telemetry_output_path}")

    telemetry_logger = TelemetryLogger(
        output_path=str(telemetry_output_path)
    )

    try:
        bot = create_bot(
            bot_type=bot_type,
            client=client,
            telemetry_logger=telemetry_logger,
        )
    except FileNotFoundError as error:
        print("\nDecision Tree model is missing.")
        print("Train it first with:")
        print("  python -m src.main train-decision-tree")
        raise error

    try:
        bot.solve(maze_name)
    except MazeApiError as error:
        if "already played this maze" in str(error).lower():
            print(
                f"Maze '{maze_name}' was already played with this token. "
                "Choose another maze or run with --reset-player."
            )
            return

        raise

    print("\nCurrent player after run:")
    print(client.get_player())

    print("\nTelemetry written to:")
    print(telemetry_output_path)


def collect_training_data(
    *,
    maze_names: list[str],
    telemetry_output_path: Path,
    fresh_telemetry: bool,
    player_name: str | None = None,
) -> None:
    """
    Collects telemetry for Decision Tree training.

    Only baseline and reward-aware bots are used here. The Decision Tree bot is
    intentionally not used for training data collection.
    """

    settings = load_settings()
    resolved_player_name = player_name or resolve_player_name(settings)

    if fresh_telemetry and telemetry_output_path.exists():
        telemetry_output_path.unlink()
        print(f"Removed existing telemetry file: {telemetry_output_path}")

    print("\nCollecting training telemetry...")
    print(f"Player: {resolved_player_name}")
    print(f"Training bots: {', '.join(TRAINING_BOT_TYPES)}")
    print(f"Training mazes: {', '.join(maze_names)}")
    print(f"Telemetry output: {telemetry_output_path}")

    for maze_name in maze_names:
        for bot_type in TRAINING_BOT_TYPES:
            print("\n" + "=" * 80)
            print(f"Collecting telemetry: bot={bot_type}, maze={maze_name}")
            print("=" * 80)

            run_bot_once(
                bot_type=bot_type,
                maze_name=maze_name,
                telemetry_output_path=telemetry_output_path,
                reset_player_on_start=True,
                player_name=resolved_player_name,
            )

    print("\nTraining telemetry collection completed.")


def run_analyze_telemetry() -> None:
    print("\nRunning telemetry analysis...")

    from src.analysis.analyze_telemetry import main as analyze_telemetry_main

    analyze_telemetry_main()

    print("Telemetry analysis completed.")
    print("Report written to: reports/telemetry_analysis.md")


def run_train_decision_tree() -> None:
    print("\nTraining Decision Tree policy...")

    from src.training.train_decision_tree import main as train_decision_tree_main

    train_decision_tree_main()

    print("Decision Tree training completed.")
    print("Model written to: models/decision_tree_policy.joblib")
    print("Training report written to: reports/decision_tree_training_report.md")


def run_evaluation(player_name: str | None = None) -> None:
    if player_name:
        os.environ["PLAYER_NAME"] = player_name

    print("\nRunning bot evaluation...")

    from src.evaluation.evaluate_bots import main as evaluate_bots_main

    evaluate_bots_main()

    print("Evaluation completed.")
    print("Evaluation report written to: reports/evaluation_report.md")


def run_pipeline(
    *,
    maze_names: list[str],
    telemetry_output_path: Path,
    fresh_telemetry: bool,
    player_name: str | None = None,
) -> None:
    print("\nStarting full pipeline...")
    print("\nPipeline steps:")
    print("  1. Collect training telemetry")
    print("  2. Analyze telemetry")
    print("  3. Train Decision Tree")
    print("  4. Evaluate all policies")
    print("  5. Produce reports")

    collect_training_data(
        maze_names=maze_names,
        telemetry_output_path=telemetry_output_path,
        fresh_telemetry=fresh_telemetry,
        player_name=player_name,
    )

    run_analyze_telemetry()
    run_train_decision_tree()
    run_evaluation(player_name=player_name)

    print(f"\nFull pipeline completed for {player_name}.")
    print("\nReview these reports:")
    print("  - reports/telemetry_analysis.md")
    print("  - reports/decision_tree_training_report.md")
    print("  - reports/evaluation_report.md")


def interactive_run_bot(player_name: str) -> None:
    maze_name = select_maze_interactively()
    bot_type = select_bot_type_interactively()
    reset_player = ask_yes_no("Reset player before run?", default=True)

    run_bot_once(
        bot_type=bot_type,
        maze_name=maze_name,
        telemetry_output_path=DEFAULT_TELEMETRY_OUTPUT_PATH,
        reset_player_on_start=reset_player,
        player_name=player_name,
    )


def interactive_collect_training_data(player_name: str) -> None:
    selected_mazes = select_multiple_mazes_interactively()
    fresh_telemetry = ask_yes_no(
        "Delete existing training telemetry before collecting new data?",
        default=True,
    )

    collect_training_data(
        maze_names=selected_mazes,
        telemetry_output_path=DEFAULT_TELEMETRY_OUTPUT_PATH,
        fresh_telemetry=fresh_telemetry,
        player_name=player_name,
    )


def interactive_pipeline(player_name: str) -> None:
    selected_mazes = select_multiple_mazes_interactively()
    fresh_telemetry = ask_yes_no(
        "Delete existing training telemetry before running the full pipeline?",
        default=True,
    )

    run_pipeline(
        maze_names=selected_mazes,
        telemetry_output_path=DEFAULT_TELEMETRY_OUTPUT_PATH,
        fresh_telemetry=fresh_telemetry,
        player_name=player_name,
    )


def show_report_paths() -> None:
    print("\nGenerated report paths:")
    print("  - reports/telemetry_analysis.md")
    print("  - reports/decision_tree_training_report.md")
    print("  - reports/decision_tree_policy.txt")
    print("  - reports/decision_tree_policy.png")
    print("  - reports/evaluation_results.csv")
    print("  - reports/evaluation_report.md")


def print_interactive_menu(player_name: str) -> None:
    print(
        f"""
Main menu

Current player: {player_name}

1. Play new game
2. Run entire pipeline with default settings
3. Register or switch player
4. List available mazes
5. Collect training telemetry
6. Analyze telemetry
7. Train Decision Tree
8. Evaluate all policies
9. Run full pipeline with custom training mazes
10. Show guide
11. Show generated report paths
0. Exit
"""
    )


def run_interactive_mode() -> None:
    settings = load_settings()
    player_name = resolve_player_name(settings)

    print_welcome(player_name)

    while True:
        print_interactive_menu(player_name)

        choice = input("Select an option: ").strip()

        try:
            if choice == "1":
                interactive_run_bot(player_name)

            elif choice == "2":
                print("\nRunning entire pipeline with default settings...")
                print(f"Player: {player_name}")
                print(f"Training mazes: {', '.join(DEFAULT_TRAINING_MAZES)}")
                print("Fresh telemetry: true")

                run_pipeline(
                    maze_names=DEFAULT_TRAINING_MAZES,
                    telemetry_output_path=DEFAULT_TELEMETRY_OUTPUT_PATH,
                    fresh_telemetry=True,
                    player_name=player_name,
                )

            elif choice == "3":
                player_name = register_new_player_interactively(player_name)

            elif choice == "4":
                list_mazes()

            elif choice == "5":
                interactive_collect_training_data(player_name)

            elif choice == "6":
                run_analyze_telemetry()

            elif choice == "7":
                run_train_decision_tree()

            elif choice == "8":
                run_evaluation(player_name=player_name)

            elif choice == "9":
                interactive_pipeline(player_name)

            elif choice == "10":
                print_guide()

            elif choice == "11":
                show_report_paths()

            elif choice == "0":
                print(f"Exiting Adaptive Maze Agent. Goodbye {player_name}!")
                break

            else:
                print("Unknown option. Please select a number from the menu.")

        except KeyboardInterrupt:
            print("\nInterrupted. Returning to main menu...")

        except Exception as error:
            print("\nAn error occurred:")
            print(error)
            print("Returning to main menu...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive Maze Agent command-line entry point."
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "guide",
        help="Show application guide and recommended workflow.",
    )

    subparsers.add_parser(
        "list-mazes",
        help="List mazes available from the API.",
    )

    run_bot_parser = subparsers.add_parser(
        "run-bot",
        help="Run one bot on one maze.",
    )
    run_bot_parser.add_argument(
        "--bot-type",
        default=None,
        help="Bot type: baseline, smart or decision_tree. Defaults to BOT_TYPE from .env.",
    )
    run_bot_parser.add_argument(
        "--maze-name",
        default=None,
        help="Maze name. Defaults to DEFAULT_MAZE_NAME from .env.",
    )
    run_bot_parser.add_argument(
        "--telemetry-path",
        default=str(DEFAULT_TELEMETRY_OUTPUT_PATH),
        help="Telemetry CSV output path.",
    )
    run_bot_parser.add_argument(
        "--reset-player",
        action="store_true",
        help="Reset player before running the bot.",
    )

    collect_parser = subparsers.add_parser(
        "collect-training-data",
        help="Collect telemetry from baseline and reward-aware bots.",
    )
    collect_parser.add_argument(
        "--mazes",
        nargs="+",
        default=DEFAULT_TRAINING_MAZES,
        help="Maze names used for training telemetry collection.",
    )
    collect_parser.add_argument(
        "--telemetry-path",
        default=str(DEFAULT_TELEMETRY_OUTPUT_PATH),
        help="Training telemetry CSV output path.",
    )
    collect_parser.add_argument(
        "--fresh-telemetry",
        action="store_true",
        help="Delete existing training telemetry before collecting new data.",
    )

    subparsers.add_parser(
        "analyze-telemetry",
        help="Generate telemetry analysis report.",
    )

    subparsers.add_parser(
        "train-decision-tree",
        help="Train the Decision Tree policy from telemetry.",
    )

    subparsers.add_parser(
        "evaluate",
        help="Evaluate all bot policies and generate evaluation reports.",
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run telemetry collection, analysis, training and evaluation.",
    )
    pipeline_parser.add_argument(
        "--mazes",
        nargs="+",
        default=DEFAULT_TRAINING_MAZES,
        help="Maze names used for training telemetry collection.",
    )
    pipeline_parser.add_argument(
        "--telemetry-path",
        default=str(DEFAULT_TELEMETRY_OUTPUT_PATH),
        help="Training telemetry CSV output path.",
    )
    pipeline_parser.add_argument(
        "--fresh-telemetry",
        action="store_true",
        help="Delete existing training telemetry before running the pipeline.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        run_interactive_mode()
        return

    if args.command == "guide":
        print_guide()
        return

    if args.command == "list-mazes":
        list_mazes()
        return

    if args.command == "run-bot":
        settings = load_settings()

        bot_type = args.bot_type or settings.bot_type
        maze_name = args.maze_name or settings.default_maze_name
        reset_player_on_start = args.reset_player or settings.reset_player_on_start
        player_name = resolve_player_name(settings)

        run_bot_once(
            bot_type=bot_type,
            maze_name=maze_name,
            telemetry_output_path=Path(args.telemetry_path),
            reset_player_on_start=reset_player_on_start,
            player_name=player_name,
        )
        return

    if args.command == "collect-training-data":
        settings = load_settings()
        player_name = resolve_player_name(settings)

        collect_training_data(
            maze_names=args.mazes,
            telemetry_output_path=Path(args.telemetry_path),
            fresh_telemetry=args.fresh_telemetry,
            player_name=player_name,
        )
        return

    if args.command == "analyze-telemetry":
        run_analyze_telemetry()
        return

    if args.command == "train-decision-tree":
        run_train_decision_tree()
        return

    if args.command == "evaluate":
        settings = load_settings()
        player_name = resolve_player_name(settings)

        run_evaluation(player_name=player_name)
        return

    if args.command == "pipeline":
        settings = load_settings()
        player_name = resolve_player_name(settings)

        run_pipeline(
            maze_names=args.mazes,
            telemetry_output_path=Path(args.telemetry_path),
            fresh_telemetry=args.fresh_telemetry,
            player_name=player_name,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()