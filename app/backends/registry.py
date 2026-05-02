"""Registry of configured backends, keyed by logical name."""

from __future__ import annotations

from app.backends.base import Backend


class BackendRegistry:
    """Holds all configured backends and owns their lifecycle."""

    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}

    def register(self, backend: Backend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"Backend {backend.name!r} already registered")
        self._backends[backend.name] = backend

    def get(self, name: str) -> Backend:
        try:
            return self._backends[name]
        except KeyError as e:
            raise KeyError(f"Unknown backend: {name!r}") from e

    def names(self) -> list[str]:
        return list(self._backends)

    def all(self) -> list[Backend]:
        return list(self._backends.values())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._backends

    async def aclose(self) -> None:
        for b in self._backends.values():
            await b.aclose()
        self._backends.clear()
