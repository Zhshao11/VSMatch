#!/usr/bin/env python3
"""E090B modality-internal structural relation descriptor diagnostic."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
ROOT=Path(os.environ.get('SPARSEGEO_ROOT','/mnt/sda/zhushihao/VS_Match'));TMP=ROOT/'image_matching_part2_migrated/.codex_tmp';RUN=ROOT/'GDROS/experiments/sparsegeo/runs/E-090B-WHU-RELATION-DESCRIPTOR';E089=ROOT/'GDROS/experiments/sparsegeo/runs/E-089A-WHU-HIGHRES-CROSSMODAL-RESIDUAL-REFINER';P0=E089/'P0'
sys.path.insert(0,str(TMP));import e089a_whu_highres_refiner as e089
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def card(s,x):d=RUN/s;dump(d/'RESULT_CARD.json',x);(d/'RESULT_CARD.md').write_text(f'# E090B {s}\n\nStatus: `{x["status"]}`\n')
def rows(n):return json.loads((E089/'P1_train/PAIR_MANIFEST.json').read_text())['pairs'][:n]
def ds(gt=True):return e089.training_dataset() if gt else e089.no_gt_dataset('training')
def batch(d,r,dev,gt=True):
 if gt:_,o,s,g,v=d[r['pair_index']];g,v=g[None].to(dev),v[None].to(dev)
 else:o,s=e089.read_images(d,r['pair_index']);g=v=None
 a=torch.load(P0/'anchor_cache'/f'{r["pair_id"]}.pt',map_location='cpu');return o[None].to(dev),s[None].to(dev),a['forward'][None].float().to(dev),g,v
class RelationNet(nn.Module):
 def __init__(self):
  super().__init__();self.opt=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,32,3,1,1));self.sar=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,32,3,1,1));self.agg=nn.Sequential(nn.Conv2d(1,12,3,1,1),nn.GELU(),nn.Conv2d(12,8,3,1,1),nn.GELU(),nn.Conv2d(8,1,1))
 def signature(self,f,img):
  f=F.normalize(f,dim=1);rels=[]
  for dy,dx in ((-2,0),(2,0),(0,-2),(0,2),(-1,-1),(-1,1),(1,-1),(1,1)):rels.append((f*torch.roll(f,(dy,dx),(2,3))).sum(1,keepdim=True))
  gray=F.interpolate(img.mean(1,keepdim=True)/255,size=f.shape[-2:],mode='bilinear',align_corners=True);gx=gray[:,:,:,2:]-gray[:,:,:,:-2];gy=gray[:,:,2:]-gray[:,:,:-2];gx=F.pad(gx,(1,1));gy=F.pad(gy,(0,0,1,1));return torch.cat(rels+[gx,gy,torch.sqrt(gx.square()+gy.square()+1e-6)],1)
 def forward(self,o,s,a):
  so=self.signature(self.opt(o/255),o);st=self.signature(self.sar(s/255),s);B,C,H,W=so.shape;ys=torch.arange(H,device=o.device,dtype=o.dtype);xs=torch.arange(W,device=o.device,dtype=o.dtype);yy,xx=torch.meshgrid(ys,xs,indexing='ij');base=torch.stack((xx,yy),-1)[None];ah=F.interpolate(a,(H,W),mode='bilinear',align_corners=True)/2;tb=base+ah.permute(0,2,3,1);cost=[]
  for dx,dy in e089.OFFSETS:
   q=e089.sample_field(st,tb+torch.tensor([dx/2,dy/2],device=o.device));cost.append(-(so-q).abs().mean(1,keepdim=True))
  raw=torch.cat(cost,1);surf=raw.permute(0,2,3,1).reshape(-1,1,9,9);ref=self.agg(surf).reshape(B,H,W,81).permute(0,3,1,2);return raw+ref
def labels(logits,a,g,v):
 r=F.interpolate(g-a,logits.shape[-2:],mode='bilinear',align_corners=True);inside=(r[:,0].abs()<=4)&(r[:,1].abs()<=4);off=torch.tensor(e089.OFFSETS,device=a.device);lab=torch.linalg.vector_norm(r.permute(0,2,3,1)[...,None,:]-off[None,None,None],dim=-1).argmin(-1);vh=F.interpolate(v[:,None].float(),lab.shape[-2:],mode='nearest')[:,0].bool();return lab,inside&vh,vh
def train(n,steps=1200):
 t=time.time();dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.manual_seed(950+n);m=RelationNet().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=8e-4);d=ds();curve=[]
 for step in range(1,steps+1):
  o,s,a,g,v=batch(d,rows(n)[(step-1)%n],dev);z=m(o,s,a);lab,mask,vh=labels(z,a,g,v);loss=F.cross_entropy(z,lab,reduction='none')[mask].mean();opt.zero_grad();loss.backward();opt.step()
  if step==1 or step%25==0:curve.append({'step':step,'ce':float(loss)});print(json.dumps(curve[-1]),flush=True)
 d0=RUN/f'{n}pair_train';d0.mkdir(parents=True,exist_ok=True);p=d0/f'E090B_{n}.pt';torch.save({'model':m.state_dict()},p);dump(d0/'CURVE.json',curve);card(f'{n}pair_train',{'status':'COMPLETE','last_completed_stage':'RELATION_CANDIDATE_TRAIN','pairs':n,'steps':steps,'checkpoint':str(p),'checkpoint_sha256':sha(p),'ce_first':curve[0]['ce'],'ce_last':curve[-1]['ce']})
def infer(n):
 dev=torch.device('cuda:0');torch.cuda.set_device(dev);tr=json.loads((RUN/f'{n}pair_train/RESULT_CARD.json').read_text());m=RelationNet().to(dev);m.load_state_dict(torch.load(tr['checkpoint'],map_location=dev)['model']);m.eval();d=ds(False);od=RUN/f'{n}pair_infer/predictions';od.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for r in rows(n):o,s,a,_,_=batch(d,r,dev,False);z=m(o,s,a);torch.save({'pair_index':r['pair_index'],'pair_id':r['pair_id'],'logits':z[0].half().cpu(),'uses_gt':False},od/f'{r["pair_id"]}.pt')
 fs=sorted(od.glob('*.pt'));man=RUN/f'{n}pair_infer/PREDICTIONS.jsonl';man.write_text(''.join(json.dumps({'path':str(p),'sha':sha(p),'uses_gt':False})+'\n' for p in fs));card(f'{n}pair_infer',{'status':'COMPLETE','last_completed_stage':'PREDICTIONS_BEFORE_GT','pairs':n,'manifest':str(man),'manifest_sha256':sha(man),'prediction_materialized_before_gt_join':True})
def join(n):
 inf=json.loads((RUN/f'{n}pair_infer/RESULT_CARD.json').read_text());recs=[json.loads(x) for x in Path(inf['manifest']).read_text().splitlines()];d=ds();t1=t5=count=0;cov=[];prs=[]
 for q in recs:
  z=torch.load(q['path']);_,o,s,g,v=d[z['pair_index']];a=torch.load(P0/'anchor_cache'/f'{z["pair_id"]}.pt')['forward'].float()[None];lab,mask,vh=labels(z['logits'][None].float(),a,g[None],v[None]);rank=z['logits'].float().argsort(0,descending=True);t1+=int((rank[0][mask[0]]==lab[0][mask[0]]).sum());t5+=int((rank[:5,mask[0]]==lab[0][mask[0]][None]).any(0).sum());count+=int(mask.sum());cov.append(float(100*mask.sum()/vh.sum()))
 top1=100*t1/count;top5=100*t5/count;gate={1:top1>=90,4:top1>=60,16:top1>=30}[n] and top1>=2*100/81
 card(f'{n}pair_join',{'status':'PASS' if gate else 'FAIL','last_completed_stage':'GT_JOIN','pairs':n,'top1_percent':top1,'top5_percent':top5,'in_window_coverage_mean':float(np.mean(cov)),'gate':gate,'prediction_materialized_before_gt_join':True})
def reduce():
 x={n:json.loads((RUN/f'{n}pair_join/RESULT_CARD.json').read_text()) for n in (1,4,16)};passed=all(v['gate'] for v in x.values());card('REDUCE',{'status':'PASS_RELATION_REPRESENTATION' if passed else 'STOP_E090B_FAILED','last_completed_stage':'PROGRESSIVE_OVERFIT_REDUCED','arms':x,'next_stage':'E090C' if passed else 'E090D_DESIGN_ONLY','whu_700_test_used':False,'os_experiment_started':False})
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['train','infer','join','reduce']);p.add_argument('--pairs',type=int);a=p.parse_args()
 try:{'train':lambda:train(a.pairs),'infer':lambda:infer(a.pairs),'join':lambda:join(a.pairs),'reduce':reduce}[a.mode]()
 except BaseException as e:card(f'{a.mode}_{a.pairs}_failure',{'status':'OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else 'IMPLEMENTATION_FAILED','last_completed_stage':a.mode,'cause':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
