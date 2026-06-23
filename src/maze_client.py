import json
from typing import Any

import requests


class MazeApiError(RuntimeError):
    pass


class MazeClient:
    def __init__(self, base_url: str, authorization_header: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": authorization_header,
                "Accept": "application/json",
            }
        )

    def _parse_response(self, response: requests.Response) -> Any:
        if response.status_code >= 400:
            raise MazeApiError(
                f"API error {response.status_code}: {response.text}"
            )

        if not response.text:
            return None

        try:
            return response.json()
        except ValueError:
            return json.loads(response.text)

    def register_player(self, name: str) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/player/register",
            params={"name": name},
        )
        return self._parse_response(response)

    def get_player(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/api/player")
        return self._parse_response(response)

    def forget_player(self) -> None:
        response = self.session.delete(f"{self.base_url}/api/player/forget")
        self._parse_response(response)

    def list_mazes(self) -> list[dict[str, Any]]:
        response = self.session.get(f"{self.base_url}/api/mazes/all")
        return self._parse_response(response)