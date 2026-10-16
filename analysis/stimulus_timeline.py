import sys,subprocess,numpy as np
W,H=64,36
raw=subprocess.run(['ffmpeg','-v','error','-i',sys.argv[1],'-vf',f'fps=10,scale={W}:{H},format=gray','-f','rawvideo','-'],capture_output=True).stdout
fr=np.frombuffer(raw,np.uint8).reshape(-1,H,W).astype(float)
dark=(fr<100).sum((1,2))
# segment by content signature
segs=[];cur=None;start=0
sig=[(d>3, None) for d in dark]
prev=None
for i,f in enumerate(fr):
    on=dark[i]>3
    key=(on, tuple(((f<100)[::6,::8]).flatten()) if on else None)
    if prev is None or key[0]!=prev[0] or (on and np.mean(np.array(key[1])!=np.array(prev[1]))>0.05):
        segs.append([i/10,on]); 
    prev=key
for a,b in zip(segs,segs[1:]+[[len(fr)/10,None]]):
    if b[0]-a[0]>=0.5: print(f'{a[0]:6.1f}-{b[0]:6.1f}s  {"STIM" if a[1] else "blank"}')
