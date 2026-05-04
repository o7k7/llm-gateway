from dataclasses import dataclass

from app.backends import BackendRegistry
from app.config import Config


@dataclass
class AppState:
    config: Config
    backends: BackendRegistry
