"""Configuration settings for the AI Receptionist application."""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8001"))
    debug: bool = False
    
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    elevenlabs_api_key: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
    
    # Twilio settings
    twilio_media_ws_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    public_ws_url: str = os.getenv("PUBLIC_WS_URL", "ws://localhost:8000/ws/twilio-media")
    
    # Socket.IO settings
    socket_cors_origins: List[str] = os.getenv("SOCKET_CORS_ORIGINS", "*").split(",")
    
    # AI Model settings
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.8"))
    whisper_model: str = os.getenv("WHISPER_MODEL", "whisper-1")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default voice
    
    # Conversation settings
    max_conversation_turns: int = int(os.getenv("MAX_CONVERSATION_TURNS", "10"))
    conversation_timeout: int = int(os.getenv("CONVERSATION_TIMEOUT", "300"))  # 5 minutes
    
    # Audio settings
    audio_sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    audio_channels: int = int(os.getenv("AUDIO_CHANNELS", "1"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        s = str(value).strip().lower()
        truthy = {"1", "true", "yes", "y", "on", "debug"}
        falsy = {"0", "false", "no", "n", "off"}
        if s in truthy:
            return True
        if s in falsy:
            return False
        # Fallback: any other string should not break; default to False
        return False


# Global settings instance
settings = Settings()
