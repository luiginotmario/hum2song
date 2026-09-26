"""Download MIR-QBSH, HumTrans, MTG-QBH, and MLEnd into the shared manifest."""

import sys

from hum2song.data.download import run_download


def main(argv: list[str] | None = None) -> None:
    """Run the downloader. MLEnd is skipped when Kaggle credentials are absent."""
    run_download(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
