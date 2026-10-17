import sys,os,csv,json,glob,numpy as np,librosa
sys.path.insert(0,'/workspace/h2s/training/src')
from multiprocessing import Pool
from hum2song.contour.features import rmvpe_contour
from hum2song.contour.evaluate import query_contour
from hum2song.contour.whistle import spectral_peak_track, WHISTLE_F0_MAX_HZ
SO='/workspace/synth_oss'; EXCL={'cl001','yt-pSQk-4fddDI'}  # MLEnd overlap: Purple Rain, Greatest Showman
M=json.load(open(SO+'/prep/meta.json'))
def pyin(f):
    y,_=librosa.load(f,sr=16000);f0,v,p=librosa.pyin(y,fmin=65,fmax=1000,sr=16000,frame_length=1024,hop_length=160)
    return np.stack([np.nan_to_num(f0),p]).astype(np.float32)
def job(a):
    kind,path,sec,out=a
    if os.path.exists(out): return out
    if kind=='ref': c=rmvpe_contour(pyin(path))
    elif kind=='whistle': y,_=librosa.load(path,sr=32000); c=query_contour(rmvpe_contour(spectral_peak_track(y),WHISTLE_F0_MAX_HZ))
    else: c=query_contour(rmvpe_contour(pyin(path)))
    np.save(out,c.astype(np.float32)); return out
jobs=[];rows=[]
for sec in M:
    song=sec.rsplit('_s',1)[0]
    if song in EXCL: continue
    ref=f'oss/refs/{sec}.npy'; jobs.append(('ref',M[sec]['src'],sec,ref))
    w=f'{SO}/whistle/whistle_{sec}.wav'
    cand=[('whistle',w)] if os.path.exists(w) else []
    cand+=[('sing',p) for p in sorted(glob.glob(f'{SO}/sing/{sec}_*.wav'))]
    for k,p in cand:
        o=f'oss/f0/{os.path.basename(p)[:-4]}.npy'; jobs.append((k,p,sec,o))
        rows.append(dict(contour=o,ref_contour=ref,song_id='oss:'+song,start_frame=0,length_frames=1000,origin='synthetic',presentation='synthetic; not a recording of a person',generator='yingmusic-singer-plus' if k=='sing' else 'ddsp-whistle',query_id=os.path.basename(p)[:-4],split='train',style=k))
if 1:
    for i,_ in enumerate(map(job,jobs)):
        if i%100==0: print(i,len(jobs),flush=True)
ok=[r for r in rows if np.load(r['contour']).shape[0]>50 and np.isfinite(np.load(r['ref_contour'])).sum()>50]
open('oss/manifest.jsonl','w').write('\n'.join(json.dumps(x) for x in ok))
print('rows',len(ok),'of',len(rows),'songs',len({r['song_id'] for r in ok}),{k:sum(r['style']==k for r in ok) for k in ('sing','whistle')})
