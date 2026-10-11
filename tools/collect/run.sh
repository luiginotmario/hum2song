#!/bin/bash
# Detached runner for the D-043 collection page (port 8300) + Cloudflare quick tunnel.
# Restarting gives a NEW trycloudflare URL (printed at the end).
DIR=/lambda/nfs/hum2song-data/dev/collect
PORT=8300
cd "$DIR"
: > tunnel.yml  # empty config: ignore the box-wide /etc/cloudflared ingress (404 catch-all)
export COMPLETION_CODE=${COMPLETION_CODE:-C1H2SPLT}

stop() { [ -f "$1" ] && kill "$(cat "$1")" 2>/dev/null; rm -f "$1"; }

stop app.pid
stop tunnel.pid
setsid nohup /lambda/nfs/hum2song-data/venv/h2s/bin/uvicorn app:app --host 127.0.0.1 --port $PORT > app.log 2>&1 < /dev/null & echo $! > app.pid
setsid nohup /usr/local/bin/cloudflared tunnel --config "$DIR/tunnel.yml" --no-autoupdate --url http://127.0.0.1:$PORT > tunnel.log 2>&1 < /dev/null & echo $! > tunnel.pid
sleep 10
grep -o "https://[a-z0-9-]*\.trycloudflare\.com" tunnel.log | head -1
