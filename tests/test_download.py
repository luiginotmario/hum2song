"""Parsers, checksummed downloads, and archive extraction. No external datasets."""

import hashlib
import io
import json
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hum2song.data.download import build_pairs, fetch_dataset, run_download
from hum2song.data.fetch import Checksum, assert_member_safe, download_file, extract_archive
from hum2song.data.sources import parse_dataset
from hum2song.data.splits import assert_no_song_leakage
from tests.support import write_quarter_note, write_tone


class RangeHandler(BaseHTTPRequestHandler):
    """Serves one payload and honors a single Range request."""

    payload = b""

    def do_GET(self) -> None:
        data = type(self).payload
        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.split("=")[1].split("-")[0])
            chunk = data[start:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        return


def test_cached_download_skips_the_network(tmp_path: Path) -> None:
    dest = tmp_path / "blob.bin"
    dest.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    status = download_file("http://127.0.0.1:9/missing", dest, Checksum("sha256", digest))
    assert status == "cached"


def test_download_resumes_a_partial_file(tmp_path: Path) -> None:
    payload = b"abcdefghijklmnopqrstuvwxyz" * 20
    RangeHandler.payload = payload
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    dest = tmp_path / "blob.bin"
    partial = tmp_path / "blob.bin.partial"
    partial.write_bytes(payload[:15])
    try:
        status = download_file(f"http://127.0.0.1:{port}/blob", dest)
    finally:
        server.shutdown()
        server.server_close()
    assert status == "resumed"
    assert dest.read_bytes() == payload


def test_extract_is_idempotent_and_rejects_parent_paths(tmp_path: Path) -> None:
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("note.txt", "one")
    dest = tmp_path / "out"
    assert extract_archive(archive, dest) == "extracted"
    (dest / "note.txt").write_text("changed", encoding="utf-8")
    assert extract_archive(archive, dest) == "cached"
    assert (dest / "note.txt").read_text(encoding="utf-8") == "changed"
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as bundle:
        bundle.writestr(zipfile.ZipInfo("../evil.txt"), "nope")
    with pytest.raises(RuntimeError):
        extract_archive(bad, tmp_path / "bad-out")
    with pytest.raises(RuntimeError):
        assert_member_safe("../evil.txt")


def test_tar_extracts_a_nested_file(tmp_path: Path) -> None:
    archive = tmp_path / "tiny.tar.gz"
    payload = b"hi"
    info = tarfile.TarInfo("dir/note.txt")
    info.size = len(payload)
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.addfile(info, io.BytesIO(payload))
    dest = tmp_path / "tar-out"
    extract_archive(archive, dest)
    assert (dest / "dir" / "note.txt").read_bytes() == payload


def test_mirqbsh_manifest_holds_every_song_out(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus" / "MIR-QBSH-corpus"
    midi_dir = corpus / "midiFile"
    midi_dir.mkdir(parents=True)
    (midi_dir / "songList.txt").write_text(
        "00001\tTwinkle, Twinkle, Little Star\t小星星\t2\n",
        encoding="utf-8",
    )
    write_quarter_note(midi_dir / "00001.mid")
    for person in ("person00001", "person00002"):
        wav = corpus / "waveFile" / "year2003" / person / "00001.wav"
        write_tone(wav, 220.0)
    data_root = tmp_path / "data"
    examples = parse_dataset("mirqbsh", [corpus])
    records, songs = build_pairs(examples, data_root, max_items=None)
    assert len(records) == 2
    assert {record.split for record in records} == {"test"}
    assert {record.song_id for record in records} == {"mirqbsh:00001"}
    assert {record.qtype for record in records} == {"hum"}
    assert records[0].song_path is not None
    assert (data_root / records[0].song_path).exists()
    assert (data_root / records[0].query_path).exists()
    assert songs[0]["title"] == "Twinkle, Twinkle, Little Star"
    assert_no_song_leakage([(record.song_id, record.split) for record in records])


def test_humtrans_keeps_the_official_split(tmp_path: Path) -> None:
    root = tmp_path / "humtrans"
    (root / "midi_data").mkdir(parents=True)
    (root / "wav_data").mkdir()
    keys = {
        "TRAIN": ["F01_0001_0001_1"],
        "VALID": ["F01_0002_0001_1"],
        "TEST": ["F01_0099_0001_1"],
    }
    (root / "train_valid_test_keys.json").write_text(json.dumps(keys), encoding="utf-8")
    for key, frequency in (
        ("F01_0001_0001_1", 220.0),
        ("F01_0002_0001_1", 330.0),
        ("F01_0099_0001_1", 440.0),
    ):
        write_tone(root / "wav_data" / f"{key}.wav", frequency)
        write_quarter_note(root / "midi_data" / f"{key}.mid")
    records, _songs = build_pairs(parse_dataset("humtrans", [root]), tmp_path / "data", None)
    splits = {record.song_id: record.split for record in records}
    assert splits == {
        "humtrans:0001": "train",
        "humtrans:0002": "val",
        "humtrans:0099": "test",
    }
    assert_no_song_leakage([(record.song_id, record.split) for record in records])


def test_mlend_rows_are_test_and_keep_hum_versus_whistle(tmp_path: Path) -> None:
    root = tmp_path / "mlend"
    audio = root / "audio"
    audio.mkdir(parents=True)
    write_tone(audio / "0000.wav", 200.0)
    write_tone(audio / "0001.wav", 400.0)
    (root / "MLEndHWD_audio_attributes.csv").write_text(
        "filename,Interpreter,Song,Interpretation\n"
        "0000.wav,1,Potter,Hum\n"
        "0001.wav,1,Potter,Whistle\n",
        encoding="utf-8",
    )
    records, _songs = build_pairs(parse_dataset("mlend", [root]), tmp_path / "data", None)
    assert {record.split for record in records} == {"test"}
    assert {record.qtype for record in records} == {"hum", "whistle"}
    assert {record.song_id for record in records} == {"mlend:Potter"}
    assert all(record.song_path is None for record in records)


def test_mlend_without_credentials_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    assert fetch_dataset("mlend", tmp_path) is None


def test_dry_run_prints_urls_without_downloading(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    run_download(["--datasets", "mirqbsh,mlend", "--out", str(tmp_path), "--dry-run"])
    assert not any(tmp_path.rglob("*"))
    assert "music-ir.org" in caplog.text


def test_max_items_keeps_each_split(tmp_path: Path) -> None:
    root = tmp_path / "humtrans"
    (root / "midi_data").mkdir(parents=True)
    (root / "wav_data").mkdir()
    keys = {
        "TRAIN": ["F01_0001_0001_1", "F01_0001_0001_2"],
        "VALID": ["F01_0002_0001_1"],
        "TEST": ["F01_0099_0001_1"],
    }
    (root / "train_valid_test_keys.json").write_text(json.dumps(keys), encoding="utf-8")
    for key in keys["TRAIN"] + keys["VALID"] + keys["TEST"]:
        write_tone(root / "wav_data" / f"{key}.wav", 250.0)
        write_quarter_note(root / "midi_data" / f"{key}.mid")
    records, _songs = build_pairs(parse_dataset("humtrans", [root]), tmp_path / "data", max_items=3)
    assert len(records) == 3
    assert {record.split for record in records} == {"train", "val", "test"}
