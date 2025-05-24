# backend/app/__init__.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, AsyncContextManager, Any

from backend.app.api.chat_routes import router as chat_router
from backend.app.api.tool_routes import router as tool_router
from backend.app.models.database import create_tables
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(lifespan: Optional[AsyncContextManager[Any]] = None) -> FastAPI:
    """Create and configure the FastAPI application"""

    # Create FastAPI app with optional lifespan
    app_kwargs = {
        "title": "Ceylon Guide Chatbot API",
        "description": "AI-powered Sri Lanka Tourism Assistant",
        "version": "1.0.0"
    }

    if lifespan is not None:
        app_kwargs["lifespan"] = lifespan

    app = FastAPI(**app_kwargs)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create database tables
    create_tables()

    # Include routers
    app.include_router(chat_router, prefix="/api")
    app.include_router(tool_router, prefix="/api")
    app.include_router(chat_router, prefix="/api/chat")

    @app.get("/")
    async def root():
        return {
            "message": "Welcome to Ceylon Guide Chatbot API",
            "version": "1.0.0",
            "endpoints": {
                "chat": "/api/chat",
                "tools": "/api/tools",
                "docs": "/docs"
            }
        }

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "Ceylon Guide Chatbot"
        }

    logger.info("Ceylon Guide Chatbot API initialized successfully")

    return app