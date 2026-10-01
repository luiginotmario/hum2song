# D-037 Classics / Purple Rain ingest — result note
Date: 2026-09-30 ET (Lambda process finished ~2026-10-01 00:14 UTC)

## Goal
Add Prince – Purple Rain plus a batch of hummable classics to the live search index (youtube_full, E2b window pipeline).

## Pipeline
1. Resolve YouTube IDs via yt-dlp search (prefer official/audio).
2. Download audio: agent box for first wave; Luigi Mac short retry for YouTube bot-blocked IPs (Lambda IP fully blocked; box rate-limited mid-batch). No Spotify.
3. `add_youtube_batch.py --name youtube_charts_v1` → `extract_previews.py` → `index_tracks.py` / `index_windows.py` with live ckpt `contour_e2b_s0/best.pt`.
4. Source tag: `youtube_full` / chart `classics_hum_d037`.

## Results
| Metric | Before | After |
|---|---:|---:|
| songs | 8001 | **8060** |
| searchable | 7245 | **7304** |
| chunks | 187570 | **189919** |
| windows | 912917 | **924305** |

- **Purple Rain** `song_id=youtube:OOM6TiWJdps` artist=Prince title=Purple Rain chunks=32 windows=144 duration_s≈245.
- D-037 rows in `youtube_charts_v1`: **59** (all searchable with chunk_count>0).
- Batch1: +31 (1 target_like drop: Simon & Garfunkel Sounds of Silence).
- Batch2: +28 (4 target_like drops among Mac retries).
- API `/health` hot-reads DB counts; SongCache loads vectors on demand — **no API restart required**.

## Added titles (59)
Africa; Alive; All Star; All the Small Things; Another Brick in the Wall; Another One Bites the Dust; Baby One More Time; Back in Black; Basket Case; Beautiful Day; Buddy Holly; Californication; Call Me Maybe; Closing Time; Come As You Are; Despacito; Dont Stop Me Now; Enter Sandman; Everlong; Free Bird; Happy; Hello; Here Comes the Sun; Hey Ya; Hotel California; House of the Rising Sun; I Want It That Way; I Will Always Love You; I Will Survive; Kiss; Like a Rolling Stone; Livin on a Prayer; Losing My Religion; Never Gonna Give You Up; Paint It Black; Piano Man; Pumped Up Kicks; Purple Rain; Respect; Rocket Man; Royals; Shake It Off; Shape of You; Somebody That I Used to Know; Somebody to Love; Superstition; Sweet Caroline; Sweet Child O Mine; Tainted Love; Take Me Home Country Roads; Thriller; Under Pressure; Under the Bridge; Wannabe; Whats Going On; When Doves Cry; With or Without You; YMCA; Your Song.

## Notable skips / failures
- Already in library before D-037: Wonderwall, Smells Like Teen Spirit, Dont Stop Believin, Take On Me, Bohemian Rhapsody, Billie Jean, etc.
- target_like drops (eval-target collision): Sounds of Silence; plus 4 from batch2 (likely Stairway / Wish You Were Here / My Way / Dont Start Now or similar — check batch_meta vs library).
- YouTube bot-check: Lambda cannot yt-dlp download; box partially works then blocks; Mac short download OK for research path.

## Ops paths
- Jobs: `/lambda/nfs/hum2song-data/jobs/d037_classics/`
- Results: this file under `/lambda/nfs/hum2song-data/results/d037/`
