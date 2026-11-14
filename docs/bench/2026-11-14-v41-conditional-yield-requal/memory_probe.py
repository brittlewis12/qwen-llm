import argparse,json,os,re,subprocess,time
from pathlib import Path

def snapshot():
    text=subprocess.check_output(['vm_stat'],text=True,timeout=5)
    page=int(re.search(r'page size of (\d+) bytes',text)[1])
    values={k:int(v) for k,v in re.findall(r'^"?([^":\n]+)"?:\s+(\d+)\.',text,re.M)}
    free=values['Pages free'];inactive=values['Pages inactive'];internal=values['Anonymous pages']
    return dict(time=time.time(),page_bytes=page,free_bytes=free*page,conservative_available_bytes=(free+max(0,inactive-internal))*page,inactive_bytes=inactive*page,anonymous_bytes=internal*page,file_backed_bytes=values['File-backed pages']*page,compressor_bytes=values['Pages occupied by compressor']*page,load=list(os.getloadavg()))

def wait_until_ready(out,deadline,minimum=40*(1<<30),stable=15):
    ready=None;last_print=0
    with Path(out).open('a') as f:
        while time.time()<deadline:
            r=snapshot();r.update(minimum_available_bytes=minimum,admission_preflight_ready=r['conservative_available_bytes']>=minimum)
            f.write(json.dumps(r)+'\n');f.flush()
            if time.time()-last_print>=50:
                print(json.dumps(r),flush=True);last_print=time.time()
            if r['admission_preflight_ready']:
                ready=ready or time.monotonic()
                if time.monotonic()-ready>=stable:return r
            else:ready=None
            time.sleep(min(10,max(0,deadline-time.time())))
    return None

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--deadline',type=float);a=p.parse_args()
    if a.deadline:
        if wait_until_ready(a.out,a.deadline) is None:raise SystemExit(75)
    else:print(json.dumps(snapshot()),flush=True)
