#!/bin/bash
# D-051: run D-049 arms A/B/C on one Lambda GPU; watchdog terminates on done/failure/stall>10min/2h cap. Never prints secrets.
cd /workspace/d049/gpu; mkdir -p logs; L(){ echo "$(date +%T) $*" >> /workspace/d049/gpu/logs/gpu.log; }
API=https://cloud.lambdalabs.com/api/v1; AU="-u $LAMBDA_API_KEY:"
K="-i $HOME/.ssh/hum2song_lambda -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30"
term(){ id=$(cat /workspace/d049/gpu/instance_id 2>/dev/null); [ -n "$id" ] && curl -s $AU -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$id\"]}" $API/instance-operations/terminate | python3 -c "import sys,json;d=json.load(sys.stdin);print('TERMINATE',[i.get('status') for i in d.get('data',{}).get('terminated_instances',[])],d.get('error',''))" >> /workspace/d049/gpu/logs/gpu.log; date +%s > /workspace/d049/gpu/t_end; sleep 30; curl -s $AU $API/instances | python3 -c "import sys,json;d=json.load(sys.stdin)['data'];print('INSTANCES_RUNNING',len([i for i in d if i['status']!='terminated']))" >> /workspace/d049/gpu/logs/gpu.log; }
T0=$(date +%s)
if [ ! -s instance_id ]; then
  for t in gpu_1x_a10 gpu_1x_a100_sxm4 gpu_1x_a100 gpu_1x_h100_pcie gpu_1x_gh200; do
    reg=$(curl -s $AU $API/instance-types | python3 -c "import sys,json;d=json.load(sys.stdin)['data'];r=d.get('$t',{}).get('regions_with_capacity_available',[]);print(r[0]['name'] if r else '')")
    [ -z "$reg" ] && continue
    r=$(curl -s $AU -H 'Content-Type: application/json' -d "{\"region_name\":\"$reg\",\"instance_type_name\":\"$t\",\"ssh_key_names\":[\"hum2song\"],\"quantity\":1,\"name\":\"d051-train\"}" $API/instance-operations/launch)
    id=$(echo "$r" | python3 -c "import sys,json;print(json.load(sys.stdin).get('data',{}).get('instance_ids',[''])[0])" 2>/dev/null)
    if [ -n "$id" ]; then echo $id > instance_id; echo $t > instance_type; date +%s > t_launch; L "launched $t $reg $id"; break; fi
    L "launch fail $t $reg $(echo $r | head -c 200)"
  done
fi
[ -s instance_id ] || { L "NO CAPACITY"; exit 1; }
trap 'L "trap -> terminate"; term' EXIT; trap 'exit 1' INT TERM HUP
id=$(cat /workspace/d049/gpu/instance_id)
for i in $(seq 60); do ip=$(curl -s $AU $API/instances/$id | python3 -c "import sys,json;d=json.load(sys.stdin)['data'];print(d.get('ip') or '' if d['status']=='active' else '')"); [ -n "$ip" ] && break; sleep 20; done
[ -z "$ip" ] && { L "never active"; exit 1; }; echo $ip > ip; H=ubuntu@$ip; L "active $ip"
for i in $(seq 20); do ssh $K $H true 2>/dev/null && break; sleep 15; done
ssh $K $H 'sudo mkdir -p /workspace && sudo chown ubuntu /workspace' || exit 1
cd /workspace
rsync -aRL -e "ssh $K" --exclude __pycache__ d049/run_d049_gpu.py d049/oss h2s/training d044/f0 h2s/docs/paper/results/d044 h2s/docs/paper/results/d050/per_clip.csv d044/data d044/ckpt/contour_e2b_s0/best.pt d047/query_synth $H:/workspace/ || { L "rsync fail"; exit 1; }
ssh $K $H 'cd /workspace/d049 && ln -sfn /workspace/d047/query_synth query_synth' 
cd /workspace/d049/gpu
ssh $K $H 'cd /workspace/d049; export PYTHONPATH=/workspace/h2s/training/src; for i in 1 2 3; do m=$(python3 -c "
import sys;sys.path.insert(0,\"/workspace/h2s/docs/paper/results/d044\")
import run_d044,hum2song.contour.train,hum2song.query_synth.dataset,hum2song.losses" 2>&1 | grep -oP "No module named .\K[a-zA-Z_0-9]+" | head -1); [ -z "$m" ] && break; echo missing $m; pip install -q $( [ $m = soundfile ] && echo soundfile==0.12.1 || echo $m ); done; python3 -c "import torch;print(torch.cuda.is_available())"' >> /workspace/d049/gpu/logs/gpu.log 2>&1
ssh $K $H 'cd /workspace/d049; PYTHONPATH=/workspace/h2s/training/src OUT=smoke MODE=mix PERT=1 timeout 900 python3 run_d049_gpu.py 10 > smoke.out 2>&1; tail -3 smoke.out; grep -q DONE smoke/run.log' >> logs/gpu.log 2>&1 || { L "SMOKE FAIL"; exit 1; }
L "smoke ok"
ssh $K $H 'cd /workspace/d049; export PYTHONPATH=/workspace/h2s/training/src;
OUT=outA MODE=mix LR=1e-5 FREEZE=0 PERT=0 NT=4 nohup python3 run_d049_gpu.py 1500 > runA.out 2>&1 &
OUT=outB MODE=oss LR=3e-6 FREEZE=0.75 PERT=0 NT=4 nohup python3 run_d049_gpu.py 1500 > runB.out 2>&1 &
OUT=outC MODE=mix LR=1e-5 FREEZE=0 PERT=1 NT=4 nohup python3 run_d049_gpu.py 1500 > runC.out 2>&1 &
sleep 2; nohup bash -c "while pgrep -f [p]ython3.run_d049_gpu >/dev/null; do sleep 15; done; touch done.flag" >/dev/null 2>&1 &'
L "A B C started"; last=""; stall=0
pull(){ rsync -a -e "ssh $K" --include 'out?/***' --include 'run?.out' --exclude '*' $H:/workspace/d049/ ./ 2>/dev/null; }
while true; do
  sleep 60
  s=$(ssh $K $H 'cd /workspace/d049; cat out?/run.log 2>/dev/null | md5sum; test -f done.flag && echo DONE' 2>/dev/null)
  pull; L "$(for x in A B C; do echo -n "$x:$(tail -1 out$x/run.log 2>/dev/null | head -c 60) | "; done) $(echo $s | grep -o DONE)"
  if echo "$s" | grep -q DONE; then pull; for x in A B C; do grep -q DONE out$x/run.log || { L "arm $x did not finish"; tail -5 run$x.out >> /workspace/d049/gpu/logs/gpu.log; }; done; L COMPLETE; break; fi
  h=$(echo "$s" | head -1); if [ -n "$h" ] && [ "$h" != "$last" ]; then last=$h; stall=0; else stall=$((stall+1)); fi
  for x in A B C; do grep -q Traceback run$x.out 2>/dev/null && { L "FAIL $x"; tail -5 run$x.out >> /workspace/d049/gpu/logs/gpu.log; exit 1; }; done
  [ $stall -ge 10 ] && { L "STALL >10min -> terminate"; exit 1; }
  [ $(( $(date +%s)-T0 )) -gt 7200 ] && { L "HARD CAP 2h -> terminate"; exit 1; }
done
