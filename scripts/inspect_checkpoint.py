import argparse,torch
p=argparse.ArgumentParser(); p.add_argument('--ckpt',required=True); a=p.parse_args()
c=torch.load(a.ckpt,map_location='cpu'); s=c.get('model',c.get('state_dict',c.get('model_state_dict',c)))
num=None
for k in ['unetpp.final.weight','final.weight']:
    if k in s: num=s[k].shape[0]; break
print('Checkpoint:',a.ckpt); print('num_classes:',num); print('handcrafted_gate:',any('handcrafted_gate' in k for k in s)); print('state keys:',len(s))
print('\nFog head keys:'); [print(' ',k) for k in s if 'fog_head' in k]
print('\nFinal keys:'); [print(' ',k,tuple(s[k].shape)) for k in s if 'final' in k]
