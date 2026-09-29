#!/bin/bash
# hum2song E2b v3 (the run that fetched most of D-028): 6 long-lived yt-dlp workers (worst audio,
# standalone deno for YouTube's JS challenges); a watcher kills them on a bot check / 429 or when a
# STOP file appears (time cutoff). Run inside ~/hum2song_yt with the standalone yt-dlp and deno.
cd "$(dirname "$0")"; export DENO_DIR="$PWD/deno_cache" DENO_NO_UPDATE_CHECK=1
RX="$(cat title_regex.txt)"; rm -f w*.urls
awk -F'\t' 'NR==FNR{d[$1]=1;next} !($1 in d){print $2 > ("w" (n++ % 6) ".urls")}' done.txt e2b_v2.tsv
for w in 0 1 2 3 4 5; do
  ./yt-dlp --no-config --no-cache-dir -i -f "wa[protocol=https]/wa/ba" --no-playlist --no-progress \
    --js-runtimes "deno:$PWD/deno" --match-filter "duration<=900 & title!~='$RX'" \
    --sleep-requests 0.5 --sleep-interval 1 --max-sleep-interval 3 \
    --print-to-file after_move:"%(id)s	%(duration)s	%(title)s" v3_meta.tsv \
    -o "audio/v3_%(id)s.%(ext)s" -a "w$w.urls" > "logs/w$w.log" 2>&1 &
done
while pgrep -f "yt-dlp --no-config" >/dev/null; do
  if grep -qi "not a bot\|HTTP Error 429\|Too Many Requests" logs/w*.log; then echo "BLOCK $(date)"; echo block > STOP; fi
  [ -e STOP ] && { pkill -f "yt-dlp --no-config"; echo "STOPPED $(cat STOP) $(date)"; break; }
  sleep 10
done
echo "V3 END $(date)"
