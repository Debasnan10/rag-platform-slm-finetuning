"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


@pytest.fixture()
def client() -> TestClient:
    """A TestClient wrapping a freshly-built FastAPI app.

    `TestClient` drives the app in-process (no real network socket),
    the same way ASP.NET Core's `WebApplicationFactory` lets you call
    endpoints directly from a test without standing up a real server.
    """

    app = create_app()
    return TestClient(app)
