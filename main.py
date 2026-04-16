"""Main FastAPI application for AI Receptionist."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ai_service.config import settings
from ai_service.services.audio_service import AudioService
from ai_service.routing.main import main_router
from main.common_settings import CORS_ALLOWED_ORIGINS

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    # Initialize services
    app.state.audio_service = AudioService()
    
    # Debug: Print all registered routes
    print("🔍 Registered routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"  {route.methods} {route.path}")
        elif hasattr(route, 'path') and hasattr(route, 'endpoint'):
            print(f"  WebSocket {route.path}")
    
    yield
    # Shutdown


# Create FastAPI app
app = FastAPI(
    title="AI Receptionist",
    description="Voice AI receptionist with Twilio, LangChain, and real-time STT/TTS",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(main_router)


if __name__ == "__main__":
    import uvicorn
    
    print("Starting server...")
    print("Server running on http://{}:{}".format(settings.host, settings.port))
    print("Host: {}".format(settings.host))
    print("Port: {}".format(settings.port))
    print("Debug: {}".format(settings.debug))
    print("Use Ctrl+C to stop the server")
    
    uvicorn.run(
        app=app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
