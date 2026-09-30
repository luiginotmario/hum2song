# hum2song

Hum, whistle, or sing a few seconds of a song and get its name back.

A fine-tuned [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) turns audio into embeddings. Songs are stored as overlapping ~10 s chunk embeddings in Postgres + pgvector. A query is embedded the same way, the nearest chunks are grouped per song, and a calibrated decision step (Jev via OpenRouter) either shows the song, asks one follow-up question, or asks you to hum again.

Status: Phase 1 (data, stage-A training, retrieval eval) is in [`training/`](training/README.md). The search API and decision layer run on the Lambda box. The one-screen hum page is in [`web/`](web/README.md). The iOS app is still a placeholder. See [`docs/SPEC.md`](docs/SPEC.md) for the build spec and [`docs/PLAN.md`](docs/PLAN.md) for research notes, datasets, and baselines.

## Hum screen

On the machine that can reach the live API (the Lambda box, or a laptop with the API tunneled to port 8000):

```bash
python web/serve.py
```

Open http://127.0.0.1:8080 . Tap the button, hum, tap again. How to tunnel and what the three results mean is in [`web/README.md`](web/README.md).

Personal research project. Model weights derived from MERT inherit its CC BY-NC 4.0 license.
