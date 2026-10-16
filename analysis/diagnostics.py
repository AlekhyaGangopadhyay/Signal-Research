import sys,glob,os,numpy as np,warnings
warnings.filterwarnings('ignore')
from scipy.signal import welch,butter,sosfiltfilt,iirnotch,filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,StratifiedKFold,GroupKFold,LeaveOneGroupOut
rng=np.random.default_rng(0)
root=sys.argv[1]
def load(f):
    s=open(f,encoding='utf-8',errors='ignore').read().split('\t')
    ser=[p for p in s if p.count(',')>=99]
    pages=[ser[i:i+20] for i in range(0,len(ser),20)]
    mats=[np.array([[float(x) for x in ch.split(',') if x.strip()] for ch in pg[:18]]) for pg in pages]
    return np.concatenate(mats,axis=1)  # 18 x total
files=sorted(glob.glob(os.path.join(root,'Raw data','*','*.txt')))
REC={}
for f in files:
    sub=os.path.basename(os.path.dirname(f)).split('(')[1].rstrip(')')
    REC[(sub,os.path.basename(f))]=load(f)
print('== 1. DATA USAGE')
tot=sum(v.shape[1] for v in REC.values()); print(f'files={len(REC)} total samples/ch={tot}  ~{tot/256/60:.1f} min @256Hz; pipeline uses 115 epochs*500={115*500} ({100*115*500/tot:.1f}%)')
for k,v in REC.items(): print(f'  {k[0]:6s} {k[1]:18s} {v.shape[1]:6d} samples = {v.shape[1]/256:6.1f}s@256 / {v.shape[1]/250:6.1f}s@250')
print('== 2. SAMPLING RATE: page=2561 samples; line-noise peak')
for fs in (250,256):
    pk=[]
    for v in list(REC.values())[:6]:
        x=v-v.mean(1,keepdims=True); fr,p=welch(x,fs=fs,nperseg=2048); m=(fr>40)&(fr<fs/2-1); pk.append(fr[m][np.argmax(p.mean(0)[m])])
    print(f'  assuming fs={fs}: peak 40Hz+ at {np.round(pk,2)}')
print('  raw quantisation: unique values ch0', len(np.unique(list(REC.values())[0][0])), 'std', list(REC.values())[0].std(1).mean().round(2))
print('== 3. SUPPORT/QUERY OVERLAP in evaluate.py sampling')
for name,counts in [('subject_3',[15]*4),('gope',[10,5,5,5]),('supra',[10,10,5,5])]:
    for k in (1,3,5):
        dup=[];
        for _ in range(2000):
            fr_=[]
            for n in counts:
                sel=rng.choice(n,size=k+5,replace=n<k+5); s,q=set(sel[:k]),sel[k:]
                fr_.append(np.mean([x in s for x in q]))
            dup.append(np.mean(fr_))
        print(f'  {name:9s} K={k}: {100*np.mean(dup):5.1f}% of query trials are identical copies of a support trial')
FS=256
def prep(x):
    sos=butter(4,[0.5,45],btype='band',fs=FS,output='sos'); x=sosfiltfilt(sos,x,axis=-1)
    b,a=iirnotch(50,30,FS); return filtfilt(b,a,x,axis=-1)
BANDS=[(1,4),(4,8),(8,13),(13,20),(20,30),(30,45)]
def feats(ep):
    fr,p=welch(ep,fs=FS,nperseg=256,axis=-1)
    return np.log(np.stack([p[:,(fr>=a)&(fr<b)].mean(-1) for a,b in BANDS],-1)+1e-12).ravel()
clf=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=3000))
print('== 4. CURRENT LABELS = FILE IDENTITY (pipeline epochs, page 1 only)')
L=['left','right','forward','backward']
for sub in ['3','Gope','Supra']:
    ks=sorted(k for k in REC if k[0].endswith(sub))
    X,yf,yd=[],[],[]
    for fi,k in enumerate(ks):
        x=prep(REC[k][:,:2561])
        for e in range(5): X.append(feats(x[:,e*500:(e+1)*500])); yf.append(fi); yd.append(fi%4)
    X,yf,yd=map(np.array,(X,yf,yd))
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    a_file=cross_val_score(clf(),X,yf,cv=cv).mean()
    a_dir_rand=cross_val_score(clf(),X,yd,cv=StratifiedKFold(5,shuffle=True,random_state=0)).mean()
    a_dir_lofo=cross_val_score(clf(),X,yd,groups=yf,cv=LeaveOneGroupOut()).mean()
    print(f'  Sub {sub:5s}: decode FILE id {100*a_file:5.1f}% (chance {100/len(ks):.0f}%) | "direction" random-split {100*a_dir_rand:5.1f}% | "direction" leave-one-file-out {100*a_dir_lofo:5.1f}% (chance 25%)')
print('== 5. CORRECTLY-LABELLED BASELINE (video timeline, onset assumed at sample 0)')
onsets=[4.2+15*i for i in range(12)]
def epochs_for(x,win=(0.5,2.5),step=None):
    out=[]
    for i,t0 in enumerate(onsets):
        d=i%4
        for s in np.arange(t0+win[0], t0+9.0-(win[1]-win[0])+1e-9, 1.0):
            a=int(s*FS); b=a+int((win[1]-win[0])*FS)
            if b<=x.shape[1]: out.append((x[:,a:b],d,i))
    return out
DS={}
for k,v in REC.items():
    x=prep(v); DS[k]=[(feats(e),d,blk) for e,d,blk in epochs_for(x)]
subs=sorted(set(k[0] for k in REC))
res={}
for sub in subs:
    ks=[k for k in DS if k[0]==sub]
    X=np.array([f for k in ks for f,_,_ in DS[k]]); y=np.array([d for k in ks for _,d,_ in DS[k]])
    g=np.array([ki for ki,k in enumerate(ks) for _ in DS[k]])
    a=cross_val_score(clf(),X,y,groups=g,cv=LeaveOneGroupOut()).mean()
    # leave-one-format-out: hold out every arrow (or letter, or word) block of every file at once.
    # (The first version grouped by (file, format) and used GroupKFold(3), which mixed formats in each fold.)
    gf=np.array([blk//4 for ki,k in enumerate(ks) for _,_,blk in DS[k]])
    a2=cross_val_score(clf(),X,y,groups=gf,cv=LeaveOneGroupOut()).mean()
    # shuffled-label control
    ys=y.copy(); 
    for gi in np.unique(g):
        m=g==gi; ys[m]=rng.permutation(ys[m])
    a3=cross_val_score(clf(),X,ys,groups=g,cv=LeaveOneGroupOut()).mean()
    res[sub]=a
    print(f'  Sub {sub:5s}: n={len(y)} epochs | leave-one-FILE-out {100*a:5.1f}% | leave-one-presentation-format-out {100*a2:5.1f}% | shuffled-label control {100*a3:5.1f}%')
for sub in subs:
    tr=[k for k in DS if k[0]!=sub]; te=[k for k in DS if k[0]==sub]
    Xtr=np.array([f for k in tr for f,_,_ in DS[k]]); ytr=np.array([d for k in tr for _,d,_ in DS[k]])
    Xte=np.array([f for k in te for f,_,_ in DS[k]]); yte=np.array([d for k in te for _,d,_ in DS[k]])
    m=clf().fit(Xtr,ytr); print(f'  LOSO target {sub:5s}: {100*m.score(Xte,yte):5.1f}%')
