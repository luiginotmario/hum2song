"""hum2song real-query collection page for Prolific (D-043).

Serves a consent -> mic check -> 10 prompts flow and stores each clip under
OWNED_ROOT/<study>/ with a per-clip JSON sidecar and an append-only manifest.csv.

Run: COMPLETION_CODE=XXXX uvicorn app:app --host 127.0.0.1 --port 8300
"""
import csv
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

HERE = Path(__file__).parent
OWNED_ROOT = Path(os.environ.get("H2S_OWNED_ROOT", "/lambda/nfs/hum2song-data/owned/prolific"))
COMPLETION_CODE = os.environ.get("COMPLETION_CODE", "CHANGE_ME")
COMPLETION_URL = "https://app.prolific.com/submissions/complete?cc={code}"
PROMPTS_PER_SESSION = int(os.environ.get("PROMPTS_PER_SESSION", "10"))
MAX_CLIP_BYTES = 8 * 1024 * 1024
MODES = {"whistle", "sing"}
EXT_BY_MIME = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/wav": "wav", "audio/x-wav": "wav"}
MANIFEST_FIELDS = [
    "clip_id", "file", "prolific_pid", "study_id", "session_id", "song_id", "artist", "title",
    "mode", "prompt_index", "attempts", "duration_s", "mime", "bytes", "client_ts", "server_ts", "user_agent",
]
SAFE_ID = re.compile(r"[^A-Za-z0-9_\-]")

SONGS = json.loads((HERE / "songs.json").read_text())
SONG_BY_ID = {s["song_id"]: s for s in SONGS}
manifest_lock = threading.Lock()
app = FastAPI(title="hum2song collect")


def safe(value, fallback="unknown"):
    cleaned = SAFE_ID.sub("", value or "")[:64]
    return cleaned or fallback


def extension(mime):
    return EXT_BY_MIME.get(mime.split(";")[0].strip().lower(), "bin")


def append_manifest(study_dir, row):
    path = study_dir / "manifest.csv"
    with manifest_lock:
        is_new = not path.exists()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/config")
def config():
    return {"songs": SONGS, "prompts": PROMPTS_PER_SESSION, "completion_url": COMPLETION_URL.format(code=COMPLETION_CODE)}


@app.get("/health")
def health():
    return {"ok": True, "songs": len(SONGS)}


@app.post("/api/clip")
async def upload_clip(
    audio: UploadFile = File(...),
    prolific_pid: str = Form(""),
    study_id: str = Form(""),
    session_id: str = Form(""),
    song_id: str = Form(...),
    mode: str = Form(...),
    prompt_index: int = Form(0),
    attempts: int = Form(1),
    duration_s: float = Form(0.0),
    client_ts: str = Form(""),
    user_agent: str = Form(""),
):
    if mode not in MODES or song_id not in SONG_BY_ID:
        raise HTTPException(400, "bad mode or song")
    data = await audio.read()
    if not data or len(data) > MAX_CLIP_BYTES:
        raise HTTPException(400, "empty or too large")

    study_dir = OWNED_ROOT / safe(study_id, "no_study")
    clip_dir = study_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    pid = safe(prolific_pid, "no_pid")
    clip_id = f"{pid}_{prompt_index:02d}_{mode}_{uuid.uuid4().hex[:8]}"
    mime = audio.content_type or "application/octet-stream"
    file_name = f"{clip_id}.{extension(mime)}"
    (clip_dir / file_name).write_bytes(data)

    song = SONG_BY_ID[song_id]
    row = {
        "clip_id": clip_id, "file": f"clips/{file_name}", "prolific_pid": pid,
        "study_id": safe(study_id, "no_study"), "session_id": safe(session_id, "no_session"),
        "song_id": song_id, "artist": song["artist"], "title": song["title"], "mode": mode,
        "prompt_index": prompt_index, "attempts": attempts, "duration_s": round(duration_s, 2),
        "mime": mime, "bytes": len(data), "client_ts": client_ts,
        "server_ts": datetime.now(timezone.utc).isoformat(), "user_agent": user_agent[:300],
    }
    (clip_dir / f"{clip_id}.json").write_text(json.dumps(row, indent=1))
    append_manifest(study_dir, row)
    return JSONResponse({"ok": True, "clip_id": clip_id})
