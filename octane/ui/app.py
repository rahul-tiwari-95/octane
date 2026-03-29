"""Octane Mission Control — FastAPI application.

Serves:
  - REST API at /api/*   (dashboard, models, traces, research, system)
  - WebSocket at /ws/*   (live events, terminal PTY)
  - React static at /*   (Vite build output)

Port: 44480 (configurable via OCTANE_UI_PORT)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from octane.ui.routes import dashboard, models, traces, research, system, queries
from octane.ui.routes.ws import router as ws_router
from octane.ui.auth import router as auth_router

logger = logging.getLogger("octane.ui")

UI_PORT = int(os.environ.get("OCTANE_UI_PORT", "44480"))

# React build output directory
_FRONTEND_DIR = Path(__file__).parent.parent.parent / "octane" / "ui-frontend" / "dist"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
        logger.info("Mission Control starting on port %d", UI_PORT)
        yield
        logger.info("Mission Control shutting down")

    app = FastAPI(
        title="Octane Mission Control",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # CORS — allow local dev server (Vite runs on 5173)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            f"http://localhost:{UI_PORT}",
            f"http://127.0.0.1:{UI_PORT}",
            "http://octane.local:5173",
            f"http://octane.local:{UI_PORT}",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(dashboard.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(traces.router, prefix="/api")
    app.include_router(research.router, prefix="/api")
    app.include_router(system.router, prefix="/api")
    app.include_router(queries.router, prefix="/api")
    app.include_router(auth_router, prefix="/api")

    # WebSocket routes
    app.include_router(ws_router)

    # Request logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info("%s %s", request.method, request.url.path)
        response = await call_next(request)
        return response

    # Serve React build if it exists
    if _FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
