"""Resumable, checksum-checked downloads and safe archive extraction."""

import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from hum2song.manifest import file_hash

USER_AGENT = "hum2song-download/0.1"
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Checksum:
    """Expected digest of a downloaded file."""

    algo: str
    value: str


def checksum_ok(path: Path, checksum: Checksum | None) -> bool:
    """True when the file matches, or when no checksum was provided."""
    if checksum is None:
        return True
    if not path.exists():
        return False
    return file_hash(path, checksum.algo) == checksum.value


def download_file(
    url: str,
    dest: Path,
    checksum: Checksum | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: int = 120,
) -> str:
    """Download url to dest. Returns 'cached', 'downloaded', or 'resumed'.

    An existing dest with a matching checksum is left untouched. A ``.partial``
    file is continued with an HTTP Range request when the server supports it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and checksum_ok(dest, checksum):
        return "cached"
    if dest.exists():
        dest.unlink()
    partial = dest.with_name(dest.name + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    if existing:
        request_headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=request_headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout_s)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"download failed for {url}: HTTP {exc.code}") from exc
    with response:
        status = getattr(response, "status", 200)
        mode, resumed_from = _partial_mode(status, existing)
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=CHUNK_BYTES)
    if not checksum_ok(partial, checksum):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {dest.name}")
    partial.replace(dest)
    if resumed_from:
        return "resumed"
    return "downloaded"


def extract_archive(archive: Path, dest: Path) -> str:
    """Extract a zip or tar into dest. Returns 'cached' when the marker matches."""
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extract_ok"
    stamp = f"{archive.stat().st_size}:{archive.stat().st_mtime_ns}"
    if marker.exists() and marker.read_text(encoding="utf-8") == stamp:
        return "cached"
    _extractor_for(archive)(archive, dest)
    marker.write_text(stamp, encoding="utf-8")
    return "extracted"


def assert_member_safe(name: str) -> None:
    """Reject absolute paths and parent-directory segments inside archives."""
    if name.startswith("/") or name.startswith("\\"):
        raise RuntimeError(f"unsafe archive member: {name}")
    if ".." in Path(name).parts:
        raise RuntimeError(f"unsafe archive member: {name}")


def _partial_mode(status: int, existing: int) -> tuple[str, int]:
    if existing and status == 206:
        return "ab", existing
    return "wb", 0


def _is_zip(path: Path) -> bool:
    return path.suffix == ".zip" or path.name.endswith(".zip")


def _extractor_for(path: Path):
    if _is_zip(path):
        return _extract_zip
    return _extract_tar


def _extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            assert_member_safe(info.filename)
        bundle.extractall(dest)


def _extract_tar_members(bundle: tarfile.TarFile, dest: Path, members: list) -> None:
    try:
        bundle.extractall(dest, members=members, filter="data")
    except TypeError:
        bundle.extractall(dest, members=members)


def _extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        for member in members:
            assert_member_safe(member.name)
        _extract_tar_members(bundle, dest, members)
