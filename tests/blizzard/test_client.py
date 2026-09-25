import base64

import httpx
import pytest

from prestie.blizzard.client import (
    BlizzardApiError,
    BlizzardNotFoundError,
    GameDataClient,
)

TOKEN_LIFETIME_S = 86399


class FakeBattleNet:
    """Stands in for oauth.battle.net and the Game Data API."""

    def __init__(self):
        self.token_requests: list[httpx.Request] = []
        self.data_requests: list[httpx.Request] = []
        self.data_responses: list[httpx.Response] = []
        self.token_status = 200
        self.issued = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth.battle.net":
            self.token_requests.append(request)
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": "invalid"})
            self.issued += 1
            return httpx.Response(
                200,
                json={
                    "access_token": f"token-{self.issued}",
                    "token_type": "bearer",
                    "expires_in": TOKEN_LIFETIME_S,
                },
            )
        self.data_requests.append(request)
        if self.data_responses:
            return self.data_responses.pop(0)
        return httpx.Response(200, json={"id": 1})


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def battle_net():
    return FakeBattleNet()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def client(battle_net, clock):
    http = httpx.Client(transport=httpx.MockTransport(battle_net.handle))
    return GameDataClient(http, "my-id", "my-secret", region="eu", clock=clock)


def test_requests_a_token_with_the_client_credentials_flow(client, battle_net):
    client.get_static("/data/wow/quest/1")

    [request] = battle_net.token_requests
    expected = base64.b64encode(b"my-id:my-secret").decode()
    assert request.method == "POST"
    assert request.headers["Authorization"] == f"Basic {expected}"
    assert request.content == b"grant_type=client_credentials"


def test_calls_the_regional_api_with_the_static_namespace(client, battle_net):
    assert client.get_static("/data/wow/quest/55763") == {"id": 1}

    [request] = battle_net.data_requests
    assert request.url.host == "eu.api.blizzard.com"
    assert request.url.path == "/data/wow/quest/55763"
    assert request.url.params["namespace"] == "static-eu"
    assert "locale" not in request.url.params  # all locales in one response
    assert request.headers["Authorization"] == "Bearer token-1"


def test_reuses_the_token_until_it_nearly_expires(client, battle_net, clock):
    client.get_static("/a")
    clock.now += TOKEN_LIFETIME_S - 120
    client.get_static("/b")
    clock.now += 90  # within the refresh margin
    client.get_static("/c")

    assert len(battle_net.token_requests) == 2
    assert battle_net.data_requests[-1].headers["Authorization"] == "Bearer token-2"


def test_a_rejected_token_is_renewed_once(client, battle_net):
    battle_net.data_responses = [httpx.Response(401), httpx.Response(200, json={})]

    assert client.get_static("/a") == {}
    assert len(battle_net.token_requests) == 2


def test_a_second_rejection_is_an_error(client, battle_net):
    battle_net.data_responses = [httpx.Response(401), httpx.Response(401)]

    with pytest.raises(BlizzardApiError, match="401"):
        client.get_static("/a")


def test_bad_credentials_are_reported(client, battle_net):
    battle_net.token_status = 401

    with pytest.raises(BlizzardApiError, match="BLIZZARD_CLIENT_ID"):
        client.get_static("/a")


def test_unknown_resource_raises_not_found(client, battle_net):
    battle_net.data_responses = [
        httpx.Response(404, json={"code": 404, "detail": "Not Found"})
    ]

    with pytest.raises(BlizzardNotFoundError):
        client.get_static("/data/wow/quest/999999999")


def test_rate_limit_is_reported(client, battle_net):
    battle_net.data_responses = [httpx.Response(429)]

    with pytest.raises(BlizzardApiError, match="rate limit"):
        client.get_static("/a")


def test_invalid_json_is_an_error(client, battle_net):
    battle_net.data_responses = [httpx.Response(200, text="<html>")]

    with pytest.raises(BlizzardApiError, match="JSON"):
        client.get_static("/a")


def test_network_failures_are_wrapped(clock):
    def fail(request):
        raise httpx.ConnectError("down", request=request)

    http = httpx.Client(transport=httpx.MockTransport(fail))
    client = GameDataClient(http, "id", "secret", region="eu", clock=clock)

    with pytest.raises(BlizzardApiError, match="reach"):
        client.get_static("/a")
