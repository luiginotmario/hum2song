# Hum screen

One page. Tap the button, hum a few seconds, tap again. The clip goes to the live search API (`POST /search?decide=true`) and the page shows one of three results: the song title, a short list to pick from, or a note to hum again.

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

Allow the microphone when the browser asks. Hum for a few seconds (it stops on its own after 12). A clear match shows the title. A close call lists the top songs. A weak or quiet hum asks you to try again.

Picking a song from the list shows that title. "Not sure" asks you to hum again. That pick stays in the browser: the API does not have a follow-up session yet.

The search API also sends open CORS headers, for a browser that calls it directly after the next uvicorn start. This page does not depend on that.
