"""
Main FastAPI application for FintelligenceAI.

This module contains the main FastAPI application instance and configuration.
"""

from contextlib import asynccontextmanager
from typing import Any

# Add DSPy import
import dspy
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import API routers
from fintelligence_ai.api.agents import router as agents_router
from fintelligence_ai.api.optimization import router as optimization_router
from fintelligence_ai.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 FintelligenceAI starting up...")

    # Configure DSPy with OpenAI
    settings = get_settings()
    if settings.openai.api_key:
        try:
            # Configure DSPy with OpenAI language model (using newer DSPy API)
            lm = dspy.LM(
                model=f"openai/{settings.openai.model}",
                api_key=settings.openai.api_key,
                temperature=settings.openai.temperature,
                max_tokens=settings.openai.max_tokens,
            )
            dspy.configure(lm=lm)
            print(f"✅ DSPy configured with OpenAI model: {settings.openai.model}")
        except Exception as e:
            print(f"⚠️ Failed to configure DSPy with OpenAI: {e}")
    else:
        print("⚠️ OpenAI API key not found - DSPy will not be configured")

    yield
    # Shutdown
    print("🛑 FintelligenceAI shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Intelligent RAG Pipeline & AI Agent System for ErgoScript generation using DSPy",
        lifespan=lifespan,
        debug=settings.app_debug,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=settings.api.cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(agents_router)
    app.include_router(optimization_router)

    return app


app = create_app()


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint with basic API information."""
    settings = get_settings()
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_environment,
        "status": "running",
        "message": "Welcome to FintelligenceAI - Intelligent RAG Pipeline & AI Agent System",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    settings = get_settings()

    # Basic health check - can be extended with dependency checks
    health_status = {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_environment,
        "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        "services": {
            "api": "healthy",
            "database": "not_checked",  # Will implement proper checks
            "redis": "not_checked",
            "chromadb": "not_checked",
        },
    }

    return health_status


@app.get("/info")
async def app_info() -> dict[str, Any]:
    """Application information endpoint."""
    settings = get_settings()

    return {
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_environment,
            "debug": settings.app_debug,
            "log_level": settings.app_log_level,
        },
        "features": {
            "rag_pipeline": "available",
            "ai_agents": "available",
            "ergoscript_generation": "available",
            "vector_search": "available",
            "document_processing": "available",
        },
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


if __name__ == "__main__":
    """Run the application directly."""
    settings = get_settings()
    uvicorn.run(
        "fintelligence_ai.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.app_log_level.lower(),
    )
