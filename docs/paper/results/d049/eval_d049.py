"""D-049/D-051 eval: GPU fine-tunes A (mix, LR1e-5), B (OSS-only, LR3e-6, 63/85 frozen), C (mix + melody-error perturbation)
vs E2b on real MLEnd test whistles: full 339, Hakuna+Potter held-out 90, D-050 P90 melody-filtered subset (304/82).
D-044 protocol; 95% clip bootstrap CIs; paired bootstrap of top-1 difference vs E2b."""
import sys,json,csv,numpy as np,torch
sys.path.insert(0,'/workspace/h2s/docs/paper/results/d044'); import run_d044 as D
from pathlib import Path
from hum2song.contour.mlend import load_contours
from hum2song.contour.data import set_input_kind
from hum2song.contour.train import load_checkpoint
DATA=Path('/workspace/d044/data'); G0='/workspace/d049/gpu'
allc,split=D.clips(DATA),D.read_split()
G=lambda s,t:[c for c in allc if split[c.person]==s and c.qtype==t]
cont={**load_contours(DATA,'rmvpe',[c for c in allc if c.qtype=='hum']),**load_contours(DATA,'peak',[c for c in allc if c.qtype=='whistle'])}
W=G('test','whistle'); H=G('test','hum')
keep={r['path']:int(r['keep_P90']) for r in csv.DictReader(open('../d050/per_clip.csv'))}
p90=np.array([keep[c.path] for c in W],bool)
hp=np.array([('Hakuna' in c.song or 'Potter' in c.song) for c in W])
CK='/workspace/d044/ckpt/contour_e2b_s0/best.pt'; dev=torch.device('cpu')
models=[('e2b',None)]+[(f'{r}_{w}',f'{G0}/out{r}/{w}_state.pt') for r in 'ABC' for w in ('best','last')]
rk={}
for name,f in models:
    m,cfg,_=load_checkpoint(CK,dev); set_input_kind(cfg.input_kind)
    if f: m.load_state_dict(torch.load(f,map_location='cpu'))
    m.eval()
    with torch.no_grad(): rk[name],pp=D.ranks(m,W,H,cont,dev)
rng=np.random.default_rng(0)
def pdiff(a,b,msk):
    ha,hb=(a[msk]<1).astype(float),(b[msk]<1).astype(float); n=len(ha)
    idx=rng.integers(0,n,(10000,n)); d=(ha[idx]-hb[idx]).mean(1)
    return {'diff':float(ha.mean()-hb.mean()),'ci95':[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]}
subs={'full':np.ones(len(W),bool),'hp90':hp,'P90':p90,'P90_hp':p90&hp}
res={}
for name in rk:
    res[name]={s:{f'top{k}':D.boot(rk[name][m],pp[m],k) for k in (1,3)} for s,m in subs.items()}
    if name!='e2b':
        for s,m in subs.items(): res[name][s]['top1_minus_e2b']=pdiff(rk[name],rk['e2b'],m)
json.dump(res,open('results.json','w'),indent=1)
with open('per_clip.csv','w') as f:
    w=csv.writer(f); w.writerow(['path','song','person','keep_P90']+[f'{n}_rank' for n in rk])
    for i,c in enumerate(W): w.writerow([c.path,c.song,c.person,int(p90[i])]+[int(rk[n][i]) for n in rk])
for name in rk:
    print(name,' | '.join(f"{s} n={res[name][s]['top1']['n']} t1 {res[name][s]['top1']['value']:.3f}[{res[name][s]['top1']['ci95_clip'][0]:.3f},{res[name][s]['top1']['ci95_clip'][1]:.3f}] t3 {res[name][s]['top3']['value']:.3f}"+(f" d {res[name][s]['top1_minus_e2b']['diff']:+.3f}[{res[name][s]['top1_minus_e2b']['ci95'][0]:+.3f},{res[name][s]['top1_minus_e2b']['ci95'][1]:+.3f}]" if name!='e2b' else '') for s in subs))
