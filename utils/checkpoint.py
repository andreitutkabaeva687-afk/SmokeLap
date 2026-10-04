from pathlib import Path
import torch

def save_checkpoint(path,model,optimizer=None,epoch=None,best_metric=None,config=None):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    d={'model':model.state_dict(),'epoch':epoch,'best_metric':best_metric,'config':config}
    if optimizer is not None: d['optimizer']=optimizer.state_dict()
    torch.save(d,p)

def load_checkpoint(path,model,device='cpu',strict=True):
    c=torch.load(path,map_location=device)
    state=c.get('model',c.get('state_dict',c.get('model_state_dict',c))) if isinstance(c,dict) else c
    return c,model.load_state_dict(state,strict=strict)
