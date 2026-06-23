import os
from dataclasses import dataclass

from dotenv import load_dotenv


AUTH_PREFIX = "HTI Thanks You"


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_token: str
    player_name: str
    default_maze_name: str

    @property
    def authorization_header(self) -> str:
        """
        The challenge API expects:
        Authorization: HTI Thanks You <TOKEN>

        To make local usage easier, MAZE_API_TOKEN can be either:
        - only the token value
        - or the full authorization header value
        """
        if self.api_token.startswith(AUTH_PREFIX):
            return self.api_token

        return f"{AUTH_PREFIX} {self.api_token}"


def load_settings() -> Settings:
    load_dotenv()

    base_url = os.getenv("MAZE_BASE_URL", "https://maze.kluster.htiprojects.nl")
    api_token = os.getenv("MAZE_API_TOKEN")
    player_name = os.getenv("PLAYER_NAME")
    default_maze_name = os.getenv("DEFAULT_MAZE_NAME", "Easy deal")

    if not api_token:
        raise ValueError("MAZE_API_TOKEN is missing. Add it to your .env file.")

    return Settings(
        base_url=base_url.rstrip("/"),
        api_token=api_token.strip(),
        player_name=(player_name or "").strip(),
        default_maze_name=default_maze_name.strip(),
    )