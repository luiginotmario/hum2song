"""Search API plus decision layer (D-017, D-027, D-032).

`mode=windows` (default) runs the D-025 window pipeline; `mode=chunks` the D-017 search.
`decide=1` on /search attaches a decision (show / ask_followup / ask_retry). POST /decide
accepts an existing search JSON (for smoke tests without re-embedding).
Browsers may call this API from another origin (CORS *). The hum screen does not
need that: `python web/serve.py` proxies /search on the same origin.

    H2S_DATABASE_URL=postgresql://... H2S_CKPT=... H2S_RMVPE=... \
        uvicorn hum2song.server.api:app --host 127.0.0.1 --port 8000
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hum2song.catalog.db import connect, has_windows, library_counts
from hum2song.catalog.search import MODES, QueryEncoder, search_audio
from hum2song.catalog.window_search import SongCache
from hum2song.decide import decide_from_search
from hum2song.decide.config import load_decide_config

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

app = FastAPI(title="hum2song search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def encoder() -> QueryEncoder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return QueryEncoder(Path(os.environ["H2S_CKPT"]), Path(os.environ["H2S_RMVPE"]), device)


@lru_cache(maxsize=1)
def database():
    return connect(os.environ["H2S_DATABASE_URL"])


@lru_cache(maxsize=1)
def song_cache() -> SongCache:
    return SongCache()


class DecideRequest(BaseModel):
    """A search payload (results + voiced_s) to classify without re-running retrieval."""

    voiced_s: float = 0.0
    results: list[dict[str, Any]] = Field(default_factory=list)
    mode: str | None = None
    message: str | None = None
    turn: int = 0


@app.get("/health")
def health() -> dict:
    counts = library_counts(database())
    decide = load_decide_config()
    return {
        "status": "ok",
        **counts,
        "window_index": has_windows(database()),
        "model": encoder().model_ver,
        "openrouter": decide.openrouter_ready,
        "decide": True,
    }


@app.post("/search")
async def search(
    audio: Annotated[UploadFile, File()],
    top_k: int = DEFAULT_TOP_K,
    mode: str = "windows",
    decide: bool = False,
    turn: int = 0,
) -> dict:
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {list(MODES)}")
    data = await audio.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio larger than 10 MB")
    suffix = Path(audio.filename or "query.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(data)
        handle.flush()
        payload = search_audio(
            encoder(),
            database(),
            Path(handle.name),
            min(top_k, MAX_TOP_K),
            mode=mode,
            cache=song_cache(),
        )
    if decide:
        payload["decision"] = decide_from_search(payload, turn=turn).as_dict()
    return payload


@app.post("/decide")
def decide_endpoint(body: DecideRequest) -> dict:
    """Classify an existing search response. Melody scores pick the branch; LLM only phrases."""
    decision = decide_from_search(
        {"voiced_s": body.voiced_s, "results": body.results},
        turn=body.turn,
    )
    return decision.as_dict()
