"""Turn an extracted dataset directory into raw query rows.

Parsers do not assign the final split and do not copy audio. Download code does that.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from hum2song.logutil import get_logger

LOGGER = get_logger(__name__)

QTYPE_FROM_MLEND = {"hum": "hum", "whistle": "whistle"}
HUMTRANS_SPLIT_NAMES = {"TRAIN": "train", "VALID": "val", "TEST": "test"}
# year2006a is "2006a-MIR補錄英文歌": the supplementary English-song session.
# The rest of MIR-QBSH does not mark humming versus singing on each clip.
MIRQBSH_SING_YEARS = frozenset({"year2006a"})


@dataclass(frozen=True)
class RawExample:
    """One recording before it is linked into the data root."""

    pair_id: str
    source_query: Path
    qtype: str
    song_id: str
    group: str
    title: str | None
    artist: str | None
    source_song: Path | None
    song_kind: str
    official_split: str | None


def parse_mirqbsh(roots: list[Path]) -> list[RawExample]:
    """MIR-QBSH: wav stem 00001-00048 is the MIDI song id.

    year2006a clips are sung English songs. Every other clip is humming,
    because the archive has no per-clip hum/sing flag.
    """
    corpus = _find_dir_with(roots, Path("midiFile") / "songList.txt")
    titles = _read_song_list(corpus / "midiFile" / "songList.txt")
    midis = {path.stem: path for path in (corpus / "midiFile").glob("*.mid")}
    examples: list[RawExample] = []
    for wav in sorted((corpus / "waveFile").rglob("*.wav")):
        song_stem = wav.stem
        person = wav.parent.name
        year = wav.parent.parent.name
        examples.append(
            RawExample(
                pair_id=f"mirqbsh:{year}:{person}:{song_stem}",
                source_query=wav,
                qtype=_mirqbsh_query_type(year),
                song_id=f"mirqbsh:{song_stem}",
                group="mirqbsh",
                title=titles.get(song_stem),
                artist=None,
                source_song=midis.get(song_stem),
                song_kind="midi" if song_stem in midis else "none",
                official_split=None,
            )
        )
    if not examples:
        raise FileNotFoundError(f"no MIR-QBSH wavs under {corpus}")
    return examples


def parse_humtrans(roots: list[Path]) -> list[RawExample]:
    """HumTrans: official split, song id is the composition (music) id, qtype hum."""
    keys_path = _find_file(roots, "train_valid_test_keys.json")
    split_of = _humtrans_splits(keys_path)
    wavs = _index_by_stem(roots, ".wav")
    midis = _index_by_stem(roots, ".mid")
    missing = 0
    examples: list[RawExample] = []
    for key in sorted(split_of):
        wav = wavs.get(key)
        if wav is None:
            missing += 1
            continue
        music_id = key.split("_")[1]
        midi = midis.get(key)
        song_kind = "midi" if midi is not None else "none"
        examples.append(
            RawExample(
                pair_id=f"humtrans:{key}",
                source_query=wav,
                qtype="hum",
                song_id=f"humtrans:{music_id}",
                group="humtrans",
                title=None,
                artist=None,
                source_song=midi,
                song_kind=song_kind,
                official_split=split_of[key],
            )
        )
    if missing:
        LOGGER.warning("HumTrans keys without a wav: %s", missing)
    if not examples:
        raise FileNotFoundError(
            "HumTrans wavs were not found. all_wav.zip has to be extracted before parsing."
        )
    return examples


def parse_mtgqbh(roots: list[Path]) -> list[RawExample]:
    """MTG-QBH sung queries. Song id is the class label (the musical piece)."""
    table = _read_csv(_find_named(roots, "queries.csv"))
    wavs = _index_by_stem(roots, ".wav")
    examples: list[RawExample] = []
    for row in table:
        filename = _column(row, "filename")
        stem = Path(filename).stem
        wav = wavs.get(stem)
        if wav is None:
            raise FileNotFoundError(f"MTG-QBH audio missing for {filename}")
        class_label = _column(row, "class label")
        query_id = _column(row, "query id") or stem
        examples.append(
            RawExample(
                pair_id=f"mtgqbh:{query_id}",
                source_query=wav,
                qtype="sing",
                song_id=f"mtgqbh:{class_label}",
                group="mtgqbh",
                title=_column(row, "title") or None,
                artist=_column(row, "original artist") or None,
                source_song=None,
                song_kind="none",
                official_split=None,
            )
        )
    if not examples:
        raise FileNotFoundError("MTG-QBH Queries.csv had no rows")
    return examples


def parse_mlend(roots: list[Path]) -> list[RawExample]:
    """MLEnd hums and whistles. All eight songs are held out later by the splitter."""
    table = _read_csv(_find_mlend_csv(roots))
    wavs = _index_by_stem(roots, ".wav")
    examples: list[RawExample] = []
    for row in table:
        filename = _column(row, "filename", "public filename")
        stem = Path(filename).stem
        wav = wavs.get(stem)
        if wav is None:
            raise FileNotFoundError(f"MLEnd audio missing for {filename}")
        song = _column(row, "song")
        interpretation = _column(row, "interpretation")
        qtype = QTYPE_FROM_MLEND.get(interpretation.lower())
        if qtype is None:
            raise ValueError(f"unknown MLEnd interpretation {interpretation!r} on {filename}")
        examples.append(
            RawExample(
                pair_id=f"mlend:{stem}",
                source_query=wav,
                qtype=qtype,
                song_id=f"mlend:{song}",
                group="mlend",
                title=song,
                artist=None,
                source_song=None,
                song_kind="none",
                official_split=None,
            )
        )
    if not examples:
        raise FileNotFoundError("MLEnd attributes CSV had no rows")
    return examples


PARSERS = {
    "mirqbsh": parse_mirqbsh,
    "humtrans": parse_humtrans,
    "mtgqbh": parse_mtgqbh,
    "mlend": parse_mlend,
}


def parse_dataset(name: str, roots: list[Path]) -> list[RawExample]:
    """Dispatch to the parser for one dataset name."""
    parser = PARSERS.get(name)
    if parser is None:
        known = ", ".join(sorted(PARSERS))
        raise ValueError(f"unknown dataset {name}. known: {known}")
    return parser(roots)


def _mirqbsh_query_type(year: str) -> str:
    if year in MIRQBSH_SING_YEARS:
        return "sing"
    return "hum"


def _humtrans_splits(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split_of: dict[str, str] = {}
    for source_name, split_name in HUMTRANS_SPLIT_NAMES.items():
        for key in payload.get(source_name, []):
            split_of[str(key)] = split_name
    return split_of


def _read_song_list(path: Path) -> dict[str, str | None]:
    titles: dict[str, str | None] = {}
    for line in _read_text(path).splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        english = parts[1].strip()
        chinese = parts[2].strip() if len(parts) > 2 else ""
        title = english if english and english != "-" else chinese
        if title == "-":
            title = ""
        titles[parts[0].strip()] = title or None
    return titles


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "big5", "cp950"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for raw in reader:
            rows.append({(key or "").strip().lower(): (value or "").strip() for key, value in raw.items()})
    return rows


def _column(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value:
            return value
    return ""


def _index_by_stem(roots: list[Path], suffix: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            if path.is_file():
                found.setdefault(path.stem, path)
    return found


def _find_file(roots: list[Path], name: str) -> Path:
    for root in roots:
        if not root.exists():
            continue
        matches = sorted(path for path in root.rglob(name) if path.is_file())
        if matches:
            return matches[0]
    raise FileNotFoundError(f"could not find {name}")


def _find_named(roots: list[Path], name: str) -> Path:
    target = name.lower()
    for root in roots:
        if not root.exists():
            continue
        matches = sorted(
            path for path in root.rglob("*") if path.is_file() and path.name.lower() == target
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(f"could not find {name}")


def _find_dir_with(roots: list[Path], relative: Path) -> Path:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(relative.name):
            parent_chain = path.relative_to(root)
            if parent_chain.as_posix().endswith(relative.as_posix()) and path.is_file():
                return path.parents[len(relative.parts) - 1]
    raise FileNotFoundError(f"could not find {relative}")


def _find_mlend_csv(roots: list[Path]) -> Path:
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        matches.extend(
            path
            for path in root.rglob("*.csv")
            if path.is_file() and "audio_attribute" in path.name.lower()
        )
    if not matches:
        raise FileNotFoundError("MLEnd audio attributes CSV was not found")
    return sorted(matches, key=_mlend_csv_rank)[0]


def _mlend_csv_rank(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    benchmark = 1 if "benchmark" in name else 0
    return (benchmark, len(name), name)
