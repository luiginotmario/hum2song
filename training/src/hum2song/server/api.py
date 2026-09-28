"""Minimal search API: POST /search with an audio file -> top-k songs (D-017).

    H2S_DATABASE_URL=postgresql://... H2S_CKPT=... H2S_RMVPE=... \\
        uvicorn hum2song.server.api:app --host 127.0.0.1 --port 8000
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from hum2song.catalog.db import connect, library_counts
from hum2song.catalog.search import QueryEncoder, search_audio

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

app = FastAPI(title="hum2song search")


@lru_cache(maxsize=1)
def encoder() -> QueryEncoder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return QueryEncoder(Path(os.environ["H2S_CKPT"]), Path(os.environ["H2S_RMVPE"]), device)


@lru_cache(maxsize=1)
def database():
    return connect(os.environ["H2S_DATABASE_URL"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **library_counts(database()), "model": encoder().model_ver}


@app.post("/search")
async def search(audio: Annotated[UploadFile, File()], top_k: int = DEFAULT_TOP_K) -> dict:
    data = await audio.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio larger than 10 MB")
    suffix = Path(audio.filename or "query.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(data)
        handle.flush()
        return search_audio(encoder(), database(), Path(handle.name), min(top_k, MAX_TOP_K))
