"""Minimal Battle.net Game Data API client (OAuth2 client credentials flow).

Client credentials in a nutshell: our app (not a player) proves who it is by
POSTing its id and secret (HTTP Basic auth) to the token endpoint, and gets a
bearer token valid ~24 h. Every API call then carries
`Authorization: Bearer <token>`. No player account is involved, so this only
opens public game data (quests, spells, items), not a character's profile.
"""

import time
from collections.abc import Callable
from typing import Any

import httpx

TOKEN_URL = "https://oauth.battle.net/token"
API_URL = "https://{region}.api.blizzard.com"
DEFAULT_TIMEOUT_S = 10.0
# Renew a bit before expiry so a token never dies between check and use.
TOKEN_REFRESH_MARGIN_S = 60


class BlizzardApiError(Exception):
    """A Blizzard API call failed; the message is safe to show."""


class BlizzardNotFoundError(BlizzardApiError):
    """The requested resource does not exist (HTTP 404)."""


def build_http_client() -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT_S)


class GameDataClient:
    def __init__(
        self,
        http: httpx.Client,
        client_id: str,
        client_secret: str,
        *,
        region: str,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._http = http
        self._credentials = (client_id, client_secret)
        self._region = region
        self._clock = clock
        self._token: str | None = None
        self._token_expires_at = 0.0

    @property
    def region(self) -> str:
        return self._region

    def get_static(self, path: str) -> dict[str, Any]:
        """GET a static game data resource (all locales in one response)."""
        params = {"namespace": f"static-{self._region}"}
        response = self._get(path, params)
        if response.status_code == httpx.codes.UNAUTHORIZED:
            self._token = None  # revoked or expired early: renew once
            response = self._get(path, params)
        return _json(response, path)

    def _get(self, path: str, params: dict[str, str]) -> httpx.Response:
        url = API_URL.format(region=self._region) + path
        headers = {"Authorization": f"Bearer {self._valid_token()}"}
        return self._send("GET", url, params=params, headers=headers)

    def _valid_token(self) -> str:
        if self._token is None or self._clock() >= self._token_expires_at:
            token, lifetime = self._request_token()
            self._token = token
            self._token_expires_at = self._clock() + lifetime - TOKEN_REFRESH_MARGIN_S
        return self._token

    def _request_token(self) -> tuple[str, float]:
        response = self._send(
            "POST",
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=self._credentials,
        )
        if response.status_code in (httpx.codes.BAD_REQUEST, httpx.codes.UNAUTHORIZED):
            raise BlizzardApiError(
                "Battle.net rejected the credentials: check BLIZZARD_CLIENT_ID "
                "and BLIZZARD_CLIENT_SECRET"
            )
        payload = _json(response, "token")
        try:
            return str(payload["access_token"]), float(payload["expires_in"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BlizzardApiError("unexpected token response from Battle.net") from exc

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise BlizzardApiError(f"cannot reach the Blizzard API: {exc}") from exc


def _json(response: httpx.Response, what: str) -> dict[str, Any]:
    if response.status_code == httpx.codes.NOT_FOUND:
        raise BlizzardNotFoundError(f"{what}: not found")
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        raise BlizzardApiError("Blizzard API rate limit reached, retry shortly")
    if response.status_code >= httpx.codes.BAD_REQUEST:
        raise BlizzardApiError(f"{what}: Blizzard API error {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise BlizzardApiError(f"{what}: invalid JSON from the Blizzard API") from exc
    if not isinstance(payload, dict):
        raise BlizzardApiError(f"{what}: unexpected JSON from the Blizzard API")
    return payload
