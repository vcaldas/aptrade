from typing import Any, Literal
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",") if i.strip()]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Use top level .env file (one level above ./backend/)
        env_file=".env",
        env_ignore_empty=True,
        extra="allow",
    )

    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    DATA_SOURCE: Path = Path("~/tradingdata/")
    PROJECT_NAME: str = "APTrade"
    # These fields are automatically loaded from .env file first, then from OS environment variables
    MASSIVE_API_KEY: str = ""
    TELEGRAM_BOTKEY: str = ""
    CHAT_ID: str = ""
    # Time zones are explicit so runtime behavior is independent of server local time.
    APP_TIMEZONE: str = "Europe/Amsterdam"
    SCANNER_TIMEZONE: str = "America/New_York"
    # Explicit event-alert controls (Option 3): business events can notify Telegram.
    ALERTS_TELEGRAM_ENABLED: bool = False
    ALERTS_TELEGRAM_EVENTS: str = "scanner_startup,scanner_window_entered,scanner_window_exited,scanner_runtime_restarted,scanner_crash"


settings = Settings()  # type: ignore
