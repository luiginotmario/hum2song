# Hum screen

One white screen. “Tap to Listen” sits above a black circle. Click it, hum a few seconds, click again. The clip goes to the live search API (`POST /search?decide=true`). The same screen then lists each match with cover art, a preview when a free 30-second clip exists, and a YouTube search link. A weak hum asks you to try again.

The browser records a WAV (the API reads audio with soundfile, which does not take MediaRecorder webm). This server serves the page and forwards `/search`, `/health`, and `/decide` to the API, so the page and the API are the same origin. The running search process does not need a restart.

## Open it

The API on the Lambda box listens on `127.0.0.1:8000`.

On that box:

```bash
python web/serve.py
```

Then open http://127.0.0.1:8080 . From a laptop, forward the page:

```bash
ssh -L 8080:127.0.0.1:8080 <lambda-host>
```

Or forward the API and run the page locally:

```bash
ssh -L 8000:127.0.0.1:8000 <lambda-host>
python web/serve.py
```

Point at another API with `--api http://127.0.0.1:8000` or `H2S_API`. `--host 0.0.0.0` listens beyond localhost. The microphone only works on localhost or https.

Allow the microphone when the browser asks. Hum for a few seconds (it stops on its own after 12). A clear match shows that song with its cover. A close call lists the top songs the same way. A weak or quiet hum asks you to try again.

Preview plays a store clip when iTunes or Deezer has one. YouTube opens a search for the title and artist in a new tab. Covers are looked up by this server at `GET /meta` (iTunes, then Deezer, then MusicBrainz and the Cover Art Archive) and cached in memory.

Picking a song title from the list shows that song. "Not sure" asks you to hum again. That pick stays in the browser: the API does not have a follow-up session yet.

The search API also sends open CORS headers, for a browser that calls it directly after the next uvicorn start. This page does not depend on that.
