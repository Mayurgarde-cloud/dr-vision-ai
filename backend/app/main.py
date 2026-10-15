"""DR Vision AI – FastAPI application entry point.

Run (from the ``backend`` folder)::

    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # makes the `ml` package importable

from . import config, database  # noqa: E402
from .routers import insights, predictions  # noqa: E402
from .services.model_service import service  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    database.init_db()
    service.load()
    yield


app = FastAPI(title="DR Vision AI", version="1.0.0",
              description="Diabetic Retinopathy severity classification using deep learning (educational use only).",
              lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

@app.get("/files/{kind}/{name}", include_in_schema=False)
def stored_file(kind: str, name: str):
    """Serve uploaded images and generated heat-maps (uuid file names, so they are cache-safe)."""
    folder = {"uploads": config.UPLOAD_DIR, "heatmaps": config.HEATMAP_DIR}.get(kind)
    if folder is None or Path(name).name != name:
        raise HTTPException(404)
    path = Path(folder) / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


app.include_router(predictions.router)
app.include_router(insights.router)

# Serve the built React app (frontend/dist) when present – lets one process host everything.
_dist = config.FRONTEND_DIST
if _dist.is_dir() and (_dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith(("api/", "files/")):
            raise HTTPException(404)
        candidate = (_dist / full_path).resolve()
        if full_path and candidate.is_file() and _dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
