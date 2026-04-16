"""Health and test routes for the AI Receptionist application."""

from fastapi import APIRouter
from datetime import datetime
from ai_service.config import settings
from receptionist.models import SystemLog

# Create router
router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint."""
    system_log = SystemLog.objects.create(
        message="Health check endpoint",
        level="info",
        created_at=datetime.now()
    )
    
    return {
        "status": "healthy",
        "service": "ai-receptionist",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/test")
async def test_endpoint():
    """Test endpoint for development."""
    return {
        "message": "AI Receptionist API is running",
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "host": settings.host,
            "port": settings.port,
            "debug": settings.debug
        }
    }


@router.get("/status")
async def status_endpoint():
    """Detailed status endpoint with service information."""
    return {
        "service": "ai-receptionist",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "host": settings.host,
            "port": settings.port,
            "debug": settings.debug,
            "openai_model": settings.openai_model,
            "whisper_model": settings.whisper_model
        },
        "endpoints": {
            "health": "/health",
            "test": "/test",
            "status": "/status",
            "twilio_webhook": "/twilio/voice",
            "websocket": "/ws/twilio-media",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }
