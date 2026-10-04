import csv
import re
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve(root, p):
    p=Path(str(p))
    return p if p.is_absolute() else Path(root)/p


def _infer_frame_id(name, fallback):
    nums=re.findall(r'\d+', Path(name).stem)
    return int(nums[-1]) if nums else int(fallback)


def smoke_level(score):
    s=float(score)
    if s<0.25: return 'L0'
    if s<0.50: return 'L1'
    if s<0.75: return 'L2'
    return 'L3'


class SmokeLapDataset(Dataset):
    REQUIRED={'image','mask','score','video_id'}
    def __init__(self,csv_file,image_dir,mask_dir,image_size=512,num_classes=9,
                 sequence_len=3,img_mean=(0.485,0.456,0.406),
                 img_std=(0.229,0.224,0.225),ignore_index=255,project_root='.',
                 augment=False,augmentation=None):
        self.project_root=Path(project_root)
        self.csv_file=_resolve(self.project_root,csv_file)
        self.image_dir=_resolve(self.project_root,image_dir)
        self.mask_dir=_resolve(self.project_root,mask_dir)
        self.image_size=int(image_size); self.num_classes=int(num_classes)
        self.sequence_len=int(sequence_len); self.ignore_index=int(ignore_index)
        self.mean=np.asarray(img_mean,dtype=np.float32)
        self.std=np.asarray(img_std,dtype=np.float32)
        self.augment=bool(augment)
        self.augmentation=dict(augmentation or {})
        if not self.csv_file.exists(): raise FileNotFoundError(f'CSV not found: {self.csv_file}')
        with self.csv_file.open('r',encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            missing=self.REQUIRED-set(reader.fieldnames or [])
            if missing: raise ValueError(f'CSV missing columns {sorted(missing)}')
            self.rows=[]
            for i,row in enumerate(reader):
                x=dict(row); x['score']=float(x['score']); x['video_id']=str(x['video_id'])
                x['frame_id']=int(x.get('frame_id') or _infer_frame_id(x['image'],i))
                self.rows.append(x)
        self.rows.sort(key=lambda x:(x['video_id'],x['frame_id']))
        by_video={}
        for i,x in enumerate(self.rows): by_video.setdefault(x['video_id'],[]).append(i)
        self.seq={}
        for ids in by_video.values():
            for pos,idx in enumerate(ids):
                s=ids[max(0,pos-self.sequence_len+1):pos+1]
                while len(s)<self.sequence_len: s.insert(0,s[0])
                self.seq[idx]=s

    def __len__(self): return len(self.rows)
    def _ip(self,p):
        p=Path(p); return p if p.is_absolute() else self.image_dir/p
    def _mp(self,p):
        p=Path(p); return p if p.is_absolute() else self.mask_dir/p
    def _load_image_array(self,p):
        path=self._ip(p); im=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if im is None: raise FileNotFoundError(f'Cannot read image: {path}')
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im=cv2.resize(im,(self.image_size,self.image_size),interpolation=cv2.INTER_LINEAR)
        return im.astype(np.float32)/255.
    def _normalize_image(self,im):
        im=(im-self.mean)/self.std
        return torch.from_numpy(im.transpose(2,0,1)).float()
    def _augment_sequence(self,images,mask):
        cfg=self.augmentation
        rng=np.random

        if rng.random() < float(cfg.get('horizontal_flip_prob',0.5)):
            images=[np.ascontiguousarray(im[:,::-1]) for im in images]
            mask=torch.flip(mask,dims=(1,))

        def draw(name,default):
            lo,hi=cfg.get(name,default)
            return float(rng.uniform(float(lo),float(hi)))

        brightness=draw('brightness',[0.75,1.25])
        contrast=draw('contrast',[0.75,1.25])
        saturation=draw('saturation',[0.70,1.30])
        gamma=draw('gamma',[0.75,1.35])
        channel_gain=rng.uniform(
            float(cfg.get('channel_gain',[0.88,1.12])[0]),
            float(cfg.get('channel_gain',[0.88,1.12])[1]),
            size=(1,1,3),
        ).astype(np.float32)
        haze=(
            float(rng.uniform(0.0,float(cfg.get('haze_strength',0.18))))
            if rng.random() < float(cfg.get('haze_prob',0.35))
            else 0.0
        )
        blur=(
            rng.random() < float(cfg.get('blur_prob',0.20))
        )
        noise_std=(
            float(rng.uniform(0.0,float(cfg.get('noise_std',0.025))))
            if rng.random() < float(cfg.get('noise_prob',0.25))
            else 0.0
        )

        augmented=[]
        for im in images:
            gray=np.sum(
                im*np.asarray([0.299,0.587,0.114],dtype=np.float32),
                axis=2,
                keepdims=True,
            )
            x=gray+saturation*(im-gray)
            center=x.mean(axis=(0,1),keepdims=True)
            x=(x-center)*contrast+center
            x=np.clip(x*brightness*channel_gain,0.0,1.0)
            x=np.power(x,gamma).astype(np.float32)
            if haze>0.0:
                haze_color=np.asarray(
                    cfg.get('haze_color',[0.82,0.84,0.86]),
                    dtype=np.float32,
                ).reshape(1,1,3)
                x=(1.0-haze)*x+haze*haze_color
            if blur:
                x=cv2.GaussianBlur(x,(3,3),sigmaX=0.0)
            if noise_std>0.0:
                x=x+rng.normal(0.0,noise_std,size=x.shape).astype(np.float32)
            augmented.append(np.clip(x,0.0,1.0).astype(np.float32))
        return augmented,mask
    def _load_mask(self,p):
        path=self._mp(p); m=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if m is None: raise FileNotFoundError(f'Cannot read mask: {path}')
        if m.ndim==3:
            if m.shape[2]==1: m=m[...,0]
            else: raise ValueError(f'Mask must be single-channel class-ID image: {path}')
        m=cv2.resize(m,(self.image_size,self.image_size),interpolation=cv2.INTER_NEAREST)
        valid=m!=self.ignore_index
        if np.any(valid):
            mn,mx=int(m[valid].min()),int(m[valid].max())
            if mn<0 or mx>=self.num_classes:
                raise ValueError(f'Mask IDs out of range in {path}: {mn}..{mx}; expected 0-{self.num_classes-1} or {self.ignore_index}')
        return torch.from_numpy(m.astype(np.int64))
    def __getitem__(self,idx):
        row=self.rows[idx]; ids=self.seq[idx]
        image_arrays=[self._load_image_array(self.rows[j]['image']) for j in ids]
        mask=self._load_mask(row['mask'])
        if self.augment:
            image_arrays,mask=self._augment_sequence(image_arrays,mask)
        images=torch.stack([self._normalize_image(im) for im in image_arrays],dim=0)
        scores=torch.tensor([self.rows[j]['score'] for j in ids],dtype=torch.float32)
        score=float(row['score'])
        return {'images':images,'mask':mask,'smoke_scores':scores,
                'smoke_score':torch.tensor(score,dtype=torch.float32),'smoke_level':smoke_level(score),
                'video_id':row['video_id'],'frame_id':row['frame_id'],'name':Path(row['image']).name}


def build_dataset(cfg,split,project_root):
    d=cfg['data']; seq=int(d.get('sequence_len',3))
    if cfg.get('model_type')=='unetpp': seq=1
    return SmokeLapDataset(d[f'{split}_csv'],d['image_dir'],d['mask_dir'],d.get('image_size',512),
        d.get('num_classes',9),seq,d.get('img_mean',[0.485,0.456,0.406]),d.get('img_std',[0.229,0.224,0.225]),
        d.get('ignore_index',255),project_root,
        augment=(split=='train' and bool(d.get('train_augmentation',{}).get('enabled',False))),
        augmentation=d.get('train_augmentation',{}))
