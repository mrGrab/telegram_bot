# telegram_bot/config.py
import sys

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError


class Settings(BaseSettings):
    """
    Application settings loaded from .env or environment variables
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        hide_input_in_errors=True,
        case_sensitive=True,
        extra="ignore",
    )

    # Core settings
    BOT_TOKEN: str
    TELEGRAM_WEBHOOK_SECRET: str = Field(
        min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_-]+$"
    )
    TELEGRAM_ALLOWED_CHAT_IDS: set[int] = Field(min_length=1)

    # Sensu settings
    SENSU_API_URL: str
    SENSU_API_KEY: str
    SENSU_NAMESPACE: str = "default"

    # Runtime defaults
    PORT: int = 80
    HOST: str = "127.0.0.1"
    DEBUG: bool = False

    # Alexa/1C Check settings
    ALEXA_HOST: str = "127.0.0.1"
    ALEXA_PORT: int = 80

    # DTEK monitoring settings
    DTEK_CITY: str = "Київ"
    DTEK_STREET: str = "Хрещатик"
    DTEK_BUILDING: str = "1"
    DTEK_STATE_FILE: str = "last_state.json"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


try:
    settings = Settings()
except (ValidationError, SettingsError, OSError, UnicodeError) as e:
    print("ERROR: Failed to load settings. Ensure required variables are set")
    print(f"Settings error: {e}")
    sys.exit(1)
