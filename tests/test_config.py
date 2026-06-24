from src.config import AUTH_PREFIX, Settings


def test_authorization_header_adds_prefix_when_only_token_is_provided():
    settings = Settings(
        base_url="https://maze.kluster.htiprojects.nl",
        api_token="abc123",
        player_name="Test Player",
        default_maze_name="Test",
    )

    assert settings.authorization_header == f"{AUTH_PREFIX} abc123"


def test_authorization_header_keeps_full_header_when_already_provided():
    full_header = f"{AUTH_PREFIX} abc123"

    settings = Settings(
        base_url="https://maze.kluster.htiprojects.nl",
        api_token=full_header,
        player_name="Test Player",
        default_maze_name="Test",
    )

    assert settings.authorization_header == full_header