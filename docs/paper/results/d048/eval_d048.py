"""D-048: CPU fine-tune of E2b on synth_oss (YingMusic-Singer sung + DDSP whistles) mixed 50/50 with the D-042
synthetic query set (query_aug as D-041/42). Select on MLEnd VAL performers (whistle->hums),
score MLEnd TEST performers (D-044 protocol; all 339 + Hakuna/Potter 90) vs E2b."""
import sys,json,time,itertools,numpy as np,torch
sys.path.insert(0,'/workspace/h2s/docs/paper/results/d044')
import run_d044 as D
from pathlib import Path
from torch.utils.data import DataLoader
from hum2song.contour.data import set_input_kind, collate_contours
from hum2song.contour.train import load_checkpoint, query_augment_spec
from hum2song.query_synth.dataset import QuerySynthDataset
from hum2song.query_synth.manifest import train_rows
from hum2song.losses import info_nce_symmetric
STEPS=int(sys.argv[1]) if len(sys.argv)>1 else 1500; B=16; OUT=Path('out'); OUT.mkdir(exist_ok=True)
torch.set_num_threads(8); dev=torch.device('cpu'); torch.manual_seed(0)
logf=open(OUT/'eval.log','a')
def log(m): print(m,flush=True); logf.write(m+'\n'); logf.flush()
CK='/workspace/d044/ckpt/contour_e2b_s0/best.pt'; DATA=Path('/workspace/d044/data')
allc,split=D.clips(DATA),D.read_split()
groups={s:{t:[c for c in allc if split[c.person]==s and c.qtype==t] for t in ('hum','whistle')} for s in ('train','val','test')}
cont={**D.load_contours(DATA,'rmvpe',[c for c in allc if c.qtype=='hum']),**D.load_contours(DATA,'peak',[c for c in allc if c.qtype=='whistle'])}
model,cfg,_=load_checkpoint(CK,dev); set_input_kind(cfg.input_kind)
cfg.query_aug=True
for k,v in dict(stretch_min=0.5,stretch_max=1.9,warp_depth=0.3,drift_semitones=1.0,jitter_semitones=0.25,dropout_prob=0.6,octave_error_prob=0.15).items(): setattr(cfg,k,v)
aug=query_augment_spec(cfg)
root=Path('.')
#el=QuerySynthDataset(root,[{**r,'song_id':'oss:'+r['ref_contour'].split('/')[-1][:-4]} for r in map(json.loads,open('oss/manifest.jsonl'))],0,refs=1,augment=aug)
#qs=QuerySynthDataset(root,train_rows(root/'query_synth/manifest.jsonl'),0,refs=1,augment=aug)
#log(f'OSS pairs {len(el)}  qsynth pairs {len(qs)}')
MAXF=600
def cap(x,rng):
    if x.shape[0]>MAXF: s=int(rng.integers(0,x.shape[0]-MAXF+1)); x=x[s:s+MAXF]
    return x
def coll(items):
    rng=np.random.default_rng()
    q,qv=collate_contours([cap(i['query'],rng) for i in items]); s,sv=collate_contours([cap(i['refs'][0],rng) for i in items])
    return dict(query=q,query_valid=qv,song=s,song_valid=sv,song_id=[i['song_id'] for i in items])
def loop(ds,seed):
    ep=0
    while True:
        ds.epoch.set(ep); yield from DataLoader(ds,batch_size=B,shuffle=True,drop_last=True,collate_fn=lambda x:x,generator=torch.Generator().manual_seed(seed+ep)); ep+=1
res={}
e2b,_,_=load_checkpoint(CK,dev); e2b.eval()
out={}
for name,f in (('e2b',None),('d048_best','best_state.pt'),('d048_last','last_state.pt')):
    m,_,_=load_checkpoint(CK,dev)
    if f: m.load_state_dict(torch.load(OUT/f))
    m.eval()
    with torch.no_grad():
        Q=D.example_set(m,groups['test']['whistle'],cont,dev); R=D.example_set(m,groups['test']['hum'],cont,dev)
        sc,ss=D.song_scores(Q,R,'centroid'); t=np.array([ss.index(x) for x in Q.songs]); rk=(sc>sc[np.arange(len(t)),t][:,None]).sum(1); pp=np.array(Q.people)
    h=np.array([('Hakuna' in x or 'Potter' in x) for x in Q.songs])
    out[name]={k:D.boot(r,p,n) for k,(r,p,n) in {'all_top1':(rk,pp,1),'all_top3':(rk,pp,3),'hp90_top1':(rk[h],pp[h],1),'hp90_top3':(rk[h],pp[h],3)}.items()}
    out[name]['n_all']=len(rk); out[name]['n_hp']=int(h.sum())
    log(name+' '+' '.join(f"{k} {v['value']:.4f} {v['ci95_clip']}" for k,v in out[name].items() if isinstance(v,dict)))
json.dump(out,open(OUT/'eval_best_last.json','w'),indent=1); log('EVALDONE')
