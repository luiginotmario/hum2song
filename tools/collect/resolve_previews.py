"""Build songs.json for the collection page: seed list + iTunes 30 s preview URLs.

Previews are linked (not hosted) from Apple's public iTunes Search API.
Run once on any machine with internet: python resolve_previews.py
"""
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
SEED = HERE / "songs_seed.tsv"
OUT = HERE / "songs.json"
ITUNES = "https://itunes.apple.com/search?media=music&entity=song&limit=5&term="
PAUSE_S = 3.2  # iTunes allows ~20 req/min


def norm(text):
    return "".join(ch for ch in text.lower() if ch.isalnum())


def search(artist, title):
    url = ITUNES + urllib.parse.quote(f"{artist} {title}")
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.load(resp).get("results", [])


def pick(results, artist, title):
    first_artist = norm(artist.split("&")[0].split(" ft.")[0])
    good = [r for r in results if first_artist in norm(r.get("artistName", "")) and r.get("previewUrl")]
    return (good or [None])[0]


def resolve(row):
    song_id, artist, title = row
    hit = pick(search(artist, title), artist, title)
    preview = hit["previewUrl"] if hit else None
    return {"song_id": song_id, "artist": artist, "title": title, "preview_url": preview}


def main():
    rows = list(csv.reader(SEED.open(), delimiter="\t"))
    songs = []
    for row in rows:
        songs.append(resolve(row))
        print(songs[-1]["title"], "->", bool(songs[-1]["preview_url"]))
        time.sleep(PAUSE_S)
    OUT.write_text(json.dumps(songs, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
