"""JSONL manifest IO shared by download, training, and eval."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

QTYPES = ("hum", "whistle", "sing")
SPLITS = ("train", "val", "test")
QSOURCES = ("real", "synth_hum", "synth_whistle", "sep_vocal", "aligned")
PAIR_FIELDS = (
    "pair_id",
    "query_path",
    "query_type",
    "qtype",
    "qsource",
    "song_id",
    "song_start_s",
    "song_dur_s",
    "split",
    "group",
    "song_path",
    "title",
)


@dataclass
class PairRecord:
    """One query-to-song row. Paths are relative to the data root."""

    pair_id: str
    query_path: str
    qtype: str
    qsource: str
    song_id: str
    song_start_s: float | None
    song_dur_s: float | None
    split: str
    group: str
    song_path: str | None = None
    title: str | None = None

    @property
    def query_type(self) -> str:
        """hum, whistle, or sing. Matching uses the melody, not the words."""
        return self.qtype


def validate_pair(record: PairRecord) -> None:
    """Raise ValueError when a row breaks the shared manifest contract."""
    if record.qtype not in QTYPES:
        raise ValueError(f"bad qtype on {record.pair_id}: {record.qtype}")
    if record.split not in SPLITS:
        raise ValueError(f"bad split on {record.pair_id}: {record.split}")
    if record.qsource not in QSOURCES:
        raise ValueError(f"bad qsource on {record.pair_id}: {record.qsource}")
    if not record.song_id or not record.query_path:
        raise ValueError(f"missing song_id or query_path on {record.pair_id}")


def pair_to_dict(record: PairRecord) -> dict:
    """Serialize a pair. query_type and qtype are the same value."""
    validate_pair(record)
    payload = asdict(record)
    payload["query_type"] = record.query_type
    return {field: payload[field] for field in PAIR_FIELDS}


def pair_from_dict(payload: dict) -> PairRecord:
    """Parse one manifest object. query_type is accepted on its own."""
    record = PairRecord(
        pair_id=str(payload["pair_id"]),
        query_path=str(payload["query_path"]),
        qtype=_resolve_query_type(payload),
        qsource=str(payload["qsource"]),
        song_id=str(payload["song_id"]),
        song_start_s=payload.get("song_start_s"),
        song_dur_s=payload.get("song_dur_s"),
        split=str(payload["split"]),
        group=str(payload["group"]),
        song_path=payload.get("song_path"),
        title=payload.get("title"),
    )
    validate_pair(record)
    return record


def _resolve_query_type(payload: dict) -> str:
    qtype = payload.get("qtype")
    query_type = payload.get("query_type")
    pair_id = payload.get("pair_id", "")
    if qtype and query_type and str(qtype) != str(query_type):
        raise ValueError(f"qtype and query_type disagree on {pair_id}")
    chosen = query_type or qtype
    if not chosen:
        raise ValueError(f"missing query_type on {pair_id}")
    return str(chosen)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomically write JSONL with sorted keys so reruns stay byte-stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file. Empty lines are ignored."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_pairs(path: Path, records: list[PairRecord]) -> None:
    """Write pair records after checking pair ids are unique."""
    seen: set[str] = set()
    rows: list[dict] = []
    for record in records:
        if record.pair_id in seen:
            raise ValueError(f"duplicate pair_id {record.pair_id}")
        seen.add(record.pair_id)
        rows.append(pair_to_dict(record))
    write_jsonl(path, rows)


def read_pairs(path: Path) -> list[PairRecord]:
    """Load pair records from JSONL."""
    return [pair_from_dict(row) for row in read_jsonl(path)]


def file_sha256(path: Path) -> str:
    """Hash a file in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        _update_digest(digest, handle)
    return digest.hexdigest()


def file_hash(path: Path, algo: str) -> str:
    """Hash a file with sha256 or md5."""
    hasher = hashlib.new(algo)
    with path.open("rb") as handle:
        _update_digest(hasher, handle)
    return hasher.hexdigest()


def _update_digest(digest, handle) -> None:
    while True:
        chunk = handle.read(1 << 20)
        if not chunk:
            return
        digest.update(chunk)
