"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


@pytest.fixture()
def client() -> TestClient:
    """A TestClient wrapping a freshly-built FastAPI app.

    `TestClient` drives the app in-process (no real network socket),
    exercising the full request/response cycle without standing up a
    real server.
    """

    app = create_app()
    return TestClient(app)
