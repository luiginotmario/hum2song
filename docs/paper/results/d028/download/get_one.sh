#!/bin/bash
# hum2song E2b v2 (earlier, slower run): the CHAD segments of one video (worst audio) via
# --download-sections; writes a STOP file on a bot check / 429. Driven by xargs -P 6.
cd "$(dirname "$0")"; export DENO_DIR="$PWD/deno_cache" DENO_NO_UPDATE_CHECK=1
tag="$1"; url="$2"; secs="$3"
[ -e STOP ] && exit 0
grep -qx "$tag" done.txt 2>/dev/null && exit 0
args=(); IFS=';' read -ra S <<< "$secs"; for s in "${S[@]}"; do args+=(--download-sections "*$s"); done
./yt-dlp --no-config --no-cache-dir -f "wa[protocol=https]/wa/ba" --no-playlist --no-progress --ffmpeg-location /opt/homebrew/bin \
  --match-filter "duration<=900 & title!~='$(cat title_regex.txt)'" --sleep-requests 0.5 --js-runtimes "deno:$PWD/deno" \
  "${args[@]}" --print-to-file after_move:"$tag	%(id)s	%(duration)s	%(title)s" e2b_meta.tsv \
  -o "audio/$tag.s%(section_start)d.%(ext)s" "$url" > "logs/$tag.log" 2>&1
cat "logs/$tag.log" >> e2b.log
if grep -qi "not a bot\|HTTP Error 429\|Too Many Requests" "logs/$tag.log"; then echo "$tag" > STOP; echo "BLOCK $tag"; fi
grep -q "does not pass filter" "logs/$tag.log" && echo "$tag" >> filtered.txt
rm -f "logs/$tag.log"; echo "$tag" >> done.txt
sleep $((1 + RANDOM % 3))
