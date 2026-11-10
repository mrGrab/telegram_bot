# telegram_bot/config.py
import sys
from typing import Literal, Self

from pydantic import BaseModel, Field, PositiveInt, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError


class TelegramChat(BaseModel):
    id: PositiveInt
    notifications: set[Literal["dtek"]] = Field(default_factory=set)


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
    TELEGRAM_CHATS: list[TelegramChat] = Field(min_length=1)

    # Home Assistant electricity check
    HASS_URL: str = Field(min_length=1)
    HASS_TOKEN: str = Field(min_length=1, repr=False)
    HASS_POWER_ENTITY_ID: str = Field(min_length=1)

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

    @model_validator(mode="after")
    def unique_chat_ids(self) -> Self:
        chat_ids = [chat.id for chat in self.TELEGRAM_CHATS]
        if len(chat_ids) != len(set(chat_ids)):
            raise ValueError("TELEGRAM_CHATS contains duplicate chat IDs")
        return self

    @property
    def allowed_chat_ids(self) -> set[int]:
        return {chat.id for chat in self.TELEGRAM_CHATS}

    def notification_chat_ids(self, notification: Literal["dtek"]) -> tuple[str, ...]:
        return tuple(
            str(chat.id)
            for chat in self.TELEGRAM_CHATS
            if notification in chat.notifications
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


try:
    settings = Settings()
except (ValidationError, SettingsError, OSError, UnicodeError) as e:
    print("ERROR: Failed to load settings. Ensure required variables are set")
    print(f"Settings error: {e}")
    sys.exit(1)
