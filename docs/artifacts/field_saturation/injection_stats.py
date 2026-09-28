import sys,torch,json
torch.set_num_threads(1)
sys.path.insert(0,'tools'); sys.path.insert(0,'.')
from chat_atom_native import load_model
from src.io.atomizer import Atomizer
import torch.nn.functional as F
m=load_model(sys.argv[1]); m.eval()
raw=open('data/corpus_fr_medium.txt','rb').read()[200000:202000].decode('utf-8','ignore')
pk=Atomizer(max_span_bytes=1).encode(raw)[:1500]
R=[];P=[];D=[]
st=m.core.state; dyn=m.core.dynamics.dynamics
with torch.no_grad():
  for p in pk:
    atom,_,_=m.compiler(p.features, atom_count=0)
    R.append(atom.r.reshape(-1)); P.append(st._projection(atom.r,atom.phi,atom.omega,atom.E,atom.kappa).reshape(-1))
    z=torch.zeros(st.n_modes,st.d_model)
    D.append((dyn(z,input_token=atom.r)).reshape(-1))
def stats(X,name):
  X=torch.stack(X); mu=X.mean(0)
  cosmu=F.cosine_similarity(X,mu.expand_as(X)).mean().item()
  frac_var=(X-mu).pow(2).sum(1).mean().item()/X.pow(2).sum(1).mean().item()
  return {name:{"rms":X.pow(2).mean().sqrt().item(),"mean_cos_to_mean":round(cosmu,4),"var_fraction(text-dependent energy share)":round(frac_var,4)}}
out={}
for X,n in ((R,'atom_r'),(P,'projection_injection'),(D,'drive_term')): out.update(stats(X,n))
print(json.dumps(out))
