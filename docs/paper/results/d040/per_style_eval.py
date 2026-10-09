# Per-style split of eval_query_synth's synthetic_holdout. Same gallery (all manifest songs x 24 windows).
import json, sys
from pathlib import Path
import numpy as np, torch
from hum2song.contour.train import load_checkpoint
from hum2song.query_synth.manifest import read_manifest
from hum2song.query_synth.report import _song_contours, _index, score_lists, query_contour
root = Path('/lambda/nfs/hum2song-data'); ckpt, out = sys.argv[1], sys.argv[2]
device = torch.device('cuda')
model, cfg, step = load_checkpoint(Path(ckpt), device)
rows = read_manifest(root / 'query_synth/manifest.jsonl')
refs, ref_ids = _index(_song_contours(root, rows), 24)
res = {'checkpoint': ckpt, 'step': step, 'origin': 'synthetic'}
for style in ('whistle', 'wrong_lyrics'):
    test = [r for r in rows if r['split'] == 'test' and r['style'] == style]
    q = [query_contour(np.load(root / r['contour'])) for r in test]
    res[style] = score_lists(model, q, [r['song_id'] for r in test], refs, ref_ids, device)
Path(out).write_text(json.dumps(res, indent=2))
print(json.dumps(res))
