"""D-050: melody-filtered MLEnd real-whistle test subset (model-independent, pre-registered).
Filter (fixed before scoring):
  C1 voiced ratio of the spectral-peak whistle track (10 ms, conf==1) >= 0.25
  C2 melody cost <= P90 of the calibration distribution.
Melody cost: key-invariant subsequence DTW of the whistle contour (100 ms, median-centred semitones,
|diff| capped at 3 st, normalised by query length) against 40 hums of the SAME song by TRAIN-split
performers (fixed seed); cost = median over the 40. Calibration: same cost for VAL-split hums vs the
same train references -> threshold = P90 (sweep P75 / P95). No model output is used.
Scoring: D-044 protocol (centroid of test hums, own performer left out); ranks computed on full test,
then subset. E2b vs D-048 last."""
import sys,json,numpy as np,torch,numba,collections
sys.path.insert(0,'/workspace/h2s/docs/paper/results/d044'); import run_d044 as D
from pathlib import Path
from hum2song.contour.mlend import load_contours
from hum2song.contour.data import set_input_kind
from hum2song.contour.train import load_checkpoint
DATA=Path('/workspace/d044/data'); OUT=Path('.'); rng=np.random.default_rng(0)
allc,split=D.clips(DATA),D.read_split()
G=lambda s,t:[c for c in allc if split[c.person]==s and c.qtype==t]
peak=np.load(DATA/'f0/peak_mlend.npz'); rm=np.load(DATA/'f0/rmvpe_mlend.npz')
cont={**load_contours(DATA,'rmvpe',[c for c in allc if c.qtype=='hum']),**load_contours(DATA,'peak',[c for c in allc if c.qtype=='whistle'])}
def seq(c):
    x=cont[c.path]; n=len(x)//5; x=np.nanmean(x[:n*5].reshape(n,5),1) if n else x
    x=x[~np.isnan(x)]; return (x-np.median(x)) if len(x) else x
@numba.njit(cache=True)
def sdtw(q,r):
    n,m=len(q),len(r); D=np.full((n+1,m+1),np.inf); D[0,:]=0.
    for i in range(1,n+1):
        for j in range(1,m+1):
            c=min(abs(q[i-1]-r[j-1]),3.0); D[i,j]=c+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[n,1:].min()/n
refs={}
for s in sorted({c.song for c in allc}):
    h=[c for c in G('train','hum') if c.song==s]; refs[s]=[seq(h[i]) for i in rng.choice(len(h),40,replace=False)]
def cost(c):
    q=seq(c)
    if len(q)<10: return 3.0
    return float(np.median([sdtw(q,r) for r in refs[c.song] if len(r)>=10]))
cal=np.array([cost(c) for c in G('val','hum')])
thr={p:float(np.percentile(cal,p)) for p in (75,90,95)}
W=G('test','whistle')
vr=np.array([float((peak[c.path][1]>0.5).mean()) if c.path in peak.files else 0. for c in W])
rmc=np.array([float(rm[c.path][1].mean()) for c in W])
mc=np.array([cost(c) for c in W])
CK='/workspace/d044/ckpt/contour_e2b_s0/best.pt'; dev=torch.device('cpu')
H=G('test','hum'); rk={}
for name,f in (('e2b',None),('d048_last','/workspace/d048/out/last_state.pt')):
    m,cfg,_=load_checkpoint(CK,dev); set_input_kind(cfg.input_kind)
    if f: m.load_state_dict(torch.load(f))
    m.eval()
    with torch.no_grad(): rk[name],pp=D.ranks(m,W,H,cont,dev)
hp=np.array([('Hakuna' in c.song or 'Potter' in c.song) for c in W])
songs=np.array([c.song for c in W])
res={'thresholds':thr,'voiced_min':0.25,'cal_n':len(cal),'subsets':{}}
masks={'full':np.ones(len(W),bool),**{f'P{p}':(vr>=0.25)&(mc<=t) for p,t in thr.items()}}
for k,msk in masks.items():
    e={'kept_per_song':dict(collections.Counter(songs[msk])),'kept':int(msk.sum())}
    for name,r in rk.items():
        for sub,sm in (('all',msk),('hp90',msk&hp)):
            for kk in (1,3): e[f'{name}_{sub}_top{kk}']=D.boot(r[sm],pp[sm],kk)
    res['subsets'][k]=e
json.dump(res,open(OUT/'results.json','w'),indent=1)
import csv
with open(OUT/'per_clip.csv','w') as f:
    w=csv.writer(f); w.writerow(['path','song','person','voiced_ratio','rmvpe_conf_mean','melody_cost','keep_P90','e2b_rank','d048_last_rank'])
    for i,c in enumerate(W): w.writerow([c.path,c.song,c.person,f'{vr[i]:.3f}',f'{rmc[i]:.3f}',f'{mc[i]:.3f}',int(masks['P90'][i]),rk['e2b'][i],rk['d048_last'][i]])
print('thr',thr)
for k,e in res['subsets'].items():
    print(k,e['kept'],{s[6:]:n for s,n in sorted(e['kept_per_song'].items())})
    for name in rk:
        print('  ',name,' '.join(f"{s}{kk} {e[f'{name}_{s}_top{kk}']['value']:.3f}[{e[f'{name}_{s}_top{kk}']['ci95_clip'][0]:.3f},{e[f'{name}_{s}_top{kk}']['ci95_clip'][1]:.3f}] n={e[f'{name}_{s}_top{kk}']['n']}" for s in ('all','hp90') for kk in (1,3)))
