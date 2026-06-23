from src.config import load_settings
from src.maze_client import MazeApiError, MazeClient
from src.models import MazeState


def print_state(state: MazeState) -> None:
    print("\nMaze state:")
    print(f"- can_collect_score_here: {state.can_collect_score_here}")
    print(f"- can_exit_maze_here: {state.can_exit_maze_here}")
    print(f"- current_score_in_hand: {state.current_score_in_hand}")
    print(f"- current_score_in_bag: {state.current_score_in_bag}")
    print(f"- tag_on_current_tile: {state.tag_on_current_tile}")
    print(f"- number_of_visits_to_tile: {state.number_of_visits_to_tile}")

    print("\nPossible moves:")
    for action in state.possible_move_actions:
        print(
            f"- {action.direction}: "
            f"reward={action.reward_on_destination}, "
            f"visited={action.has_been_visited}, "
            f"visits={action.number_of_visits_to_tile}, "
            f"exit={action.allows_exit}, "
            f"collection={action.allows_score_collection}, "
            f"tag={action.tag_on_tile}"
        )


def main() -> None:
    settings = load_settings()

    client = MazeClient(
        base_url=settings.base_url,
        authorization_header=settings.authorization_header,
    )

    try:
        client.register_player(settings.player_name)
        print("Player registered successfully.")
    except MazeApiError as error:
        error_message = str(error)

        if "already" in error_message.lower() or "409" in error_message:
            print("Player already registered. Continuing...")
        else:
            raise

    maze_name = "Test"
    print(f"\nEntering maze: {maze_name}")
    state = client.enter_maze(maze_name)
    print_state(state)

    if state.possible_move_actions:
        first_action = state.possible_move_actions[0]
        print(f"\nTrying one move: {first_action.direction}")
        state = client.move(first_action.direction)
        print_state(state)

    print("\nCurrent player after test:")
    print(client.get_player())


if __name__ == "__main__":
    main()