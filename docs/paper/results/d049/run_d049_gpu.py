"""D-049 (from D-048 template): CPU fine-tune of E2b on synth_oss+synth_oss2 (OSS sung+whistle, 50% of every batch), lr 1e-5; resumable. Was: D-048: CPU fine-tune of E2b on synth_oss (YingMusic-Singer sung + DDSP whistles) mixed 50/50 with the D-042
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
STEPS=int(sys.argv[1]) if len(sys.argv)>1 else 1500; B=16; import os; OUT=Path(os.environ.get('OUT','out')); OUT.mkdir(exist_ok=True)
torch.set_num_threads(int(os.environ.get('NT','8'))); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(0)
logf=open(OUT/'run.log','a'); LR=float(os.environ.get('LR','1e-5')); MODE=os.environ.get('MODE','mix'); FREEZE=float(os.environ.get('FREEZE','0'))
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
el=QuerySynthDataset(root,[{**r,'song_id':'oss:'+r['ref_contour'].split('/')[-1][:-4]} for r in map(json.loads,open('oss/manifest.jsonl')) if os.environ.get('PERT','0')=='1' or not r.get('pert')],0,refs=1,augment=aug)
qs=QuerySynthDataset(root,train_rows(root/'query_synth/manifest.jsonl'),0,refs=1,augment=aug)
log(f'PERT {os.environ.get("PERT","0")} OSS pairs {len(el)}  qsynth pairs {len(qs)}')
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
ie,iq=loop(el,0),loop(qs,1)
PS=list(model.parameters()); nf=int(len(PS)*FREEZE)
for p in PS[:nf]: p.requires_grad_(False)
log(f'MODE {MODE} LR {LR} froze {nf}/{len(PS)} tensors')
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=0.05); start=1
if (OUT/'ckpt.pt').exists():
    C=torch.load(OUT/'ckpt.pt',map_location=dev); model.load_state_dict(C['model']); opt.load_state_dict(C['opt']); best=tuple(C['best']); start=C['step']+1; torch.manual_seed(C['step'])
    ie,iq=loop(el,C['step']),loop(qs,C['step']+7); log(f'resumed at step {C["step"]} best {best}')
else:
    model.eval(); v=D.val_score(model,groups,cont,dev); best=(v,0); torch.save(model.state_dict(),OUT/'best_state.pt'); log(f'step 0 val {v:.4f}')
for step in range(start,STEPS+1):
    model.train(); b=coll(next(ie)+next(iq)) if MODE=='mix' else coll(next(ie)+next(ie))
    for g in opt.param_groups: g['lr']=LR*min(1.0,step/50)
    b={k:(v.to(dev,non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}
    q=model(b['query'],b['query_valid']); s=model(b['song'],b['song_valid'])
    loss=info_nce_symmetric(q.float(),s.float(),b['song_id'],model.temperature())
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step%250==0 or step==STEPS:
        model.eval(); v=D.val_score(model,groups,cont,dev); log(f'step {step} loss {float(loss.detach()):.3f} val {v:.4f}')
        torch.save(model.state_dict(),OUT/'last_state.pt')
        if v>best[0]: best=(v,step); torch.save(model.state_dict(),OUT/'best_state.pt')
        torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),step=step,best=list(best)),OUT/'ckpt.pt.tmp'); import os; os.replace(OUT/'ckpt.pt.tmp',OUT/'ckpt.pt')
log(f'best val {best[0]:.4f} step {best[1]}')
res={}
e2b,_,_=load_checkpoint(CK,dev); e2b.eval()
model.load_state_dict(torch.load(OUT/'best_state.pt',map_location=dev)); model.eval()
def sub(rk,pp,songs,keep):
    m=np.array([s in keep for s in songs]); return D.boot(rk[m],pp[m],1),D.boot(rk[m],pp[m],3)
last,_,_=load_checkpoint(CK,dev); last.load_state_dict(torch.load(OUT/'last_state.pt',map_location=dev)); last.eval()
for name,m in (('e2b',e2b),('d049_best',model),('d049_last',last)):
    with torch.no_grad():
        Q=D.example_set(m,groups['test']['whistle'],cont,dev); R=D.example_set(m,groups['test']['hum'],cont,dev)
        sc,ss=D.song_scores(Q,R,'centroid'); t=np.array([ss.index(x) for x in Q.songs]); rk=(sc>sc[np.arange(len(t)),t][:,None]).sum(1); pp=np.array(Q.people)
    h=np.array([('Hakuna' in x or 'Potter' in x) for x in Q.songs])
    o=np.array([not any(k in x for k in ('Hakuna','Potter','Rain','Showman')) for x in Q.songs])
    import csv as _c; K90={x['path'] for x in _c.DictReader(open('/workspace/h2s/docs/paper/results/d050/per_clip.csv')) if x['keep_P90']=='1'}; k9=np.array([q in K90 for q in Q.paths]) if hasattr(Q,'paths') else np.array([c.path in K90 for c in groups['test']['whistle']])
    res[name]={'p90_top1':D.boot(rk[k9],pp[k9],1),'p90hp_top1':D.boot(rk[k9&h],pp[k9&h],1),'n_p90':int(k9.sum()),'nonoverlap_top1':D.boot(rk[o],pp[o],1),'n_nonoverlap':int(o.sum()),'all8_top1':D.boot(rk,pp,1),'all8_top3':D.boot(rk,pp,3),'hp90_top1':D.boot(rk[h],pp[h],1),'hp90_top3':D.boot(rk[h],pp[h],3),'ranks':rk.tolist(),'songs':list(Q.songs)}
    log(f"{name} all8 top1 {res[name]['all8_top1']['value']:.4f} {res[name]['all8_top1']['ci95_clip']} top3 {res[name]['all8_top3']['value']:.4f} | HP n={int(h.sum())} top1 {res[name]['hp90_top1']['value']:.4f} {res[name]['hp90_top1']['ci95_clip']} | nonoverlap n={int(o.sum())} top1 {res[name]['nonoverlap_top1']['value']:.4f} {res[name]['nonoverlap_top1']['ci95_clip']}")
json.dump({'best':best,'results':res},open(OUT/'results.json','w'),indent=1)
cfg_save={'model':model.state_dict(),'note':'D-049 E2b + synth_oss+synth_oss2/qsynth CPU finetune lr1e-5'}
torch.save(cfg_save,OUT/'d049_best_raw.pt')
log('DONE')
