"""Browsers on another origin can call the search API (D-034)."""

from fastapi.testclient import TestClient

from hum2song.server import api


def test_search_allows_any_browser_origin():
    response = TestClient(api.app).options(
        "/search",
        headers={
            "Origin": "http://127.0.0.1:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "POST" in response.headers["access-control-allow-methods"]
