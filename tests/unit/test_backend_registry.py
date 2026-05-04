"""Tests for BackendRegistry — registration, lookup, lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from app.backends.registry import BackendRegistry
from app.schemas.chat import ChatChunk, ChatRequest


class _FakeBackend:
    """Minimal structural match for the Backend Protocol."""

    def __init__(self, name: str, model: str = "fake-model") -> None:
        self.name = name
        self.model = model
        self.closed = False

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        self.closed = True


class TestRegistration:
    def test_register_and_get(self) -> None:
        reg = BackendRegistry()
        b = _FakeBackend("small")
        reg.register(b)
        assert reg.get("small") is b

    def test_duplicate_register_raises(self) -> None:
        reg = BackendRegistry()
        reg.register(_FakeBackend("small"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_FakeBackend("small"))

    def test_get_missing_raises_keyerror(self) -> None:
        reg = BackendRegistry()
        with pytest.raises(KeyError, match="Unknown backend"):
            reg.get("nope")

    def test_contains(self) -> None:
        reg = BackendRegistry()
        reg.register(_FakeBackend("small"))
        assert "small" in reg
        assert "large" not in reg
        assert 42 not in reg  # non-str

    def test_names_and_all(self) -> None:
        reg = BackendRegistry()
        a = _FakeBackend("a")
        b = _FakeBackend("b")
        reg.register(a)
        reg.register(b)
        assert set(reg.names()) == {"a", "b"}
        assert set(reg.all()) == {a, b}


class TestLifecycle:
    async def test_aclose_closes_all_and_clears(self) -> None:
        reg = BackendRegistry()
        a = _FakeBackend("a")
        b = _FakeBackend("b")
        reg.register(a)
        reg.register(b)

        await reg.aclose()

        assert a.closed is True
        assert b.closed is True
        assert reg.names() == []

    async def test_aclose_swallows_individual_failures(self) -> None:
        """A flaky backend must not prevent other backends from closing or
        block the shutdown path."""
        reg = BackendRegistry()

        flaky = _FakeBackend("flaky")
        flaky.aclose = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        good = _FakeBackend("good")

        reg.register(flaky)
        reg.register(good)

        # Must NOT raise — errors logged, shutdown continues
        await reg.aclose()

        assert good.closed is True
        assert reg.names() == []
