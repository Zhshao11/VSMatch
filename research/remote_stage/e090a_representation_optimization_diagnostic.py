#!/usr/bin/env python3
"""E090A representation/optimization decoupling diagnostics."""
from __future__ import annotations
import argparse, hashlib, json, math, os, signal, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(os.environ.get('SPARSEGEO_ROOT','/mnt/sda/zhushihao/VS_Match')); TMP=ROOT/'image_matching_part2_migrated/.codex_tmp'
RUN=ROOT/'GDROS/experiments/sparsegeo/runs/E-090A-WHU-REPRESENTATION-OPTIMIZATION-DECOUPLING'
E089=ROOT/'GDROS/experiments/sparsegeo/runs/E-089A-WHU-HIGHRES-CROSSMODAL-RESIDUAL-REFINER'; P0=E089/'P0'
EXPECTED='4122c27a7a0a1d0cf6488b9207277b0024d2fe0a9f07f6fcd8eec8ef09ca745a'
sys.path.insert(0,str(TMP)); import e089a_whu_highres_refiner as e089

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def card(stage,x):
 d=RUN/stage;dump(d/'RESULT_CARD.json',x);(d/'RESULT_CARD.md').write_text(f'# E090A {stage}\n\nStatus: `{x.get("status")}`\n\nLast stage: `{x.get("last_completed_stage")}`\n')
def rows(n):return json.loads((E089/'P1_train/PAIR_MANIFEST.json').read_text())['pairs'][:n]
def dataset(gt=True):return e089.training_dataset() if gt else e089.no_gt_dataset('training')
def batch(ds,row,device,with_gt=True):
 if with_gt:
  _,o,s,g,v=ds[row['pair_index']];g,v=g[None].to(device),v[None].to(device)
 else:o,s=e089.read_images(ds,row['pair_index']);g=v=None
 a=torch.load(P0/'anchor_cache'/f'{row["pair_id"]}.pt',map_location='cpu')
 return o[None].to(device),s[None].to(device),a['forward'][None].float().to(device),a['reverse'][None].float().to(device),g,v
def labels(aux,anchor,truth,valid):
 r=F.interpolate(truth-anchor,size=aux['logits'].shape[-2:],mode='bilinear',align_corners=True)
 inside=(r[:,0].abs()<=4)&(r[:,1].abs()<=4)
 off=aux_model_offsets(aux['logits'].device)
 lab=torch.linalg.vector_norm(r.permute(0,2,3,1)[...,None,:]-off[None,None,None],dim=-1).argmin(-1)
 vh=F.interpolate(valid[:,None].float(),size=lab.shape[-2:],mode='nearest')[:,0].bool();return lab,inside&vh,vh
def aux_model_offsets(device):return torch.tensor(e089.OFFSETS,dtype=torch.float32,device=device)

def a1_train(n,steps):
 t=time.time();dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.manual_seed(900+n);m=e089.HighResolutionResidualRefiner().to(dev)
 params=list(m.opt_stem.parameters())+list(m.sar_stem.parameters())+list(m.shared_projection.parameters())+list(m.score_head.parameters())+[m.log_temperature]
 opt=torch.optim.AdamW(params,lr=8e-4,weight_decay=1e-5);ds=dataset();curve=[];hist=torch.zeros(81,dtype=torch.long);inside_n=valid_n=0
 for step in range(1,steps+1):
  o,s,a,r,g,v=batch(ds,rows(n)[(step-1)%n],dev);_,x=m(o,s,a,r);lab,mask,vh=labels(x,a,g,v);loss=F.cross_entropy(x['logits'],lab,reduction='none')[mask].mean()
  opt.zero_grad(set_to_none=True);loss.backward();grads={k:float(sum((p.grad.abs().sum() for p in ps if p.grad is not None),torch.tensor(0.,device=dev))) for k,ps in {'opt':m.opt_stem.parameters(),'sar':m.sar_stem.parameters(),'proj':m.shared_projection.parameters(),'score':m.score_head.parameters()}.items()};opt.step()
  hist+=torch.bincount(lab[mask].detach().cpu(),minlength=81);inside_n+=int(mask.sum());valid_n+=int(vh.sum())
  if step==1 or step%25==0:curve.append({'step':step,'ce':float(loss),'lr':opt.param_groups[0]['lr'],'gradients':grads});print(json.dumps(curve[-1]),flush=True)
 d=RUN/f'A1_{n}pair_train';d.mkdir(parents=True,exist_ok=True);p=d/f'A1_{n}pair.pt';torch.save({'model':m.state_dict(),'pairs':n,'steps':steps},p);dump(d/'CURVE.json',curve)
 card(f'A1_{n}pair_train',{'status':'COMPLETE','last_completed_stage':'CANDIDATE_ONLY_TRAIN','pairs':n,'steps':steps,'checkpoint':str(p),'checkpoint_sha256':sha(p),'ce_first':curve[0]['ce'],'ce_last':curve[-1]['ce'],'in_window_coverage_percent':100*inside_n/valid_n,'label_histogram':hist.tolist(),'candidate_gpus':None,'logical_device':'cuda:0','elapsed_seconds':time.time()-t})

def a1_infer(n):
 t=time.time();dev=torch.device('cuda:0');torch.cuda.set_device(dev);m=e089.HighResolutionResidualRefiner().to(dev);tr=json.loads((RUN/f'A1_{n}pair_train/RESULT_CARD.json').read_text());m.load_state_dict(torch.load(tr['checkpoint'],map_location=dev)['model']);m.eval();ds=dataset(False);d=RUN/f'A1_{n}pair_inference/predictions';d.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for row in rows(n):
   o,s,a,r,_,_=batch(ds,row,dev,False);_,x=m(o,s,a,r);torch.save({'pair_index':row['pair_index'],'pair_id':row['pair_id'],'logits':x['logits'][0].half().cpu(),'entropy':x['entropy'][0].half().cpu(),'uses_gt':False},d/f'{row["pair_id"]}.pt')
 files=sorted(d.glob('*.pt'));man=RUN/f'A1_{n}pair_inference/PREDICTIONS.jsonl';man.write_text(''.join(json.dumps({'path':str(p),'sha256':sha(p),'uses_gt':False})+'\n' for p in files));card(f'A1_{n}pair_inference',{'status':'COMPLETE','last_completed_stage':'PREDICTIONS_BEFORE_GT','pairs':n,'manifest':str(man),'manifest_sha256':sha(man),'prediction_materialized_before_gt_join':True,'elapsed_seconds':time.time()-t})

class OracleGate(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,a,r):
  ah=F.interpolate(a,scale_factor=.5,mode='bilinear',align_corners=True);rh=F.interpolate(r,scale_factor=.5,mode='bilinear',align_corners=True);cy=torch.linalg.vector_norm(ah+rh,dim=1,keepdim=True)/16;ne=torch.linalg.vector_norm(ah-F.avg_pool2d(ah,3,1,1),dim=1,keepdim=True)/8;mag=torch.linalg.vector_norm(ah,dim=1,keepdim=True)/32;return torch.sigmoid(self.net(torch.cat((cy,ne,mag),1)))
def oracle_residual(a,g):return (g-a).round().clamp(-4,4)
def a2_train(n,steps):
 t=time.time();dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.manual_seed(920+n);m=OracleGate().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=1e-3);ds=dataset();curve=[]
 for step in range(1,steps+1):
  o,s,a,r,g,v=batch(ds,rows(n)[(step-1)%n],dev);gate=F.interpolate(m(a,r),size=a.shape[-2:],mode='bilinear',align_corners=True);prop=oracle_residual(a,g);out=a+gate*prop;loss=torch.sqrt(e089.valid_epe(out,g,v).square()+.01).mean();opt.zero_grad();loss.backward();opt.step()
  if step==1 or step%25==0:curve.append({'step':step,'loss':float(loss),'gate_mean':float(gate.mean()),'aepe':float(e089.valid_epe(out,g,v).mean())})
 d=RUN/f'A2_{n}pair_train';d.mkdir(parents=True,exist_ok=True);p=d/f'A2_{n}pair.pt';torch.save({'model':m.state_dict()},p);dump(d/'CURVE.json',curve);card(f'A2_{n}pair_train',{'status':'COMPLETE','last_completed_stage':'ORACLE_PROPOSAL_GATE_TRAIN','pairs':n,'steps':steps,'checkpoint':str(p),'checkpoint_sha256':sha(p),'curve':curve,'elapsed_seconds':time.time()-t})
def a2_infer(n):
 t=time.time();dev=torch.device('cuda:0');torch.cuda.set_device(dev);m=OracleGate().to(dev);tr=json.loads((RUN/f'A2_{n}pair_train/RESULT_CARD.json').read_text());m.load_state_dict(torch.load(tr['checkpoint'],map_location=dev)['model']);m.eval();ds=dataset(False);d=RUN/f'A2_{n}pair_inference/predictions';d.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for row in rows(n):
   o,s,a,r,_,_=batch(ds,row,dev,False);gate=F.interpolate(m(a,r),size=a.shape[-2:],mode='bilinear',align_corners=True);torch.save({'pair_index':row['pair_index'],'pair_id':row['pair_id'],'anchor':a[0].half().cpu(),'gate':gate[0].half().cpu(),'uses_gt':False},d/f'{row["pair_id"]}.pt')
 files=sorted(d.glob('*.pt'));man=RUN/f'A2_{n}pair_inference/PREDICTIONS.jsonl';man.write_text(''.join(json.dumps({'path':str(p),'sha256':sha(p),'uses_gt':False})+'\n' for p in files));card(f'A2_{n}pair_inference',{'status':'COMPLETE','last_completed_stage':'GATE_MAPS_BEFORE_GT','pairs':n,'manifest':str(man),'manifest_sha256':sha(man),'prediction_materialized_before_gt_join':True,'oracle_proposal_not_materialized':True,'elapsed_seconds':time.time()-t})

def metrics(vals):
 a=np.asarray(vals);return {'pairs':len(vals),'aepe':float(a.mean()),'cmr1':float(100*(a<1).mean()),'cmr2':float(100*(a<2).mean()),'cmr5':float(100*(a<5).mean())}
def join_arm(kind,n):
 t=time.time();inf=json.loads((RUN/f'{kind}_{n}pair_inference/RESULT_CARD.json').read_text());man=Path(inf['manifest']);recs=[json.loads(x) for x in man.read_text().splitlines()];assert len(recs)==n and sha(man)==inf['manifest_sha256'];ds=dataset();out=[];top1=top5=count=0;ents=[]
 for rec in recs:
  z=torch.load(rec['path'],map_location='cpu');_,o,s,g,v=ds[z['pair_index']];a=torch.load(P0/'anchor_cache'/f'{z["pair_id"]}.pt',map_location='cpu')['forward'].float()[None];g=g[None];v=v[None]
  if kind=='A1':
   logits=z['logits'].float();dummy={'logits':logits[None]};lab,mask,vh=labels(dummy,a,g,v);rank=logits.argsort(0,descending=True);top1+=int((rank[0][mask[0]]==lab[0][mask[0]]).sum());top5+=int((rank[:5,mask[0]]==lab[0][mask[0]][None]).any(0).sum());count+=int(mask.sum());ents.append(float(z['entropy'].float().mean()));out.append({'pair_id':z['pair_id'],'top1':float(100*(rank[0][mask[0]]==lab[0][mask[0]]).float().mean()),'top5':float(100*(rank[:5,mask[0]]==lab[0][mask[0]][None]).any(0).float().mean()),'entropy':ents[-1],'in_window_percent':float(100*mask.sum()/vh.sum())})
  else:
   gate=z['gate'].float()[None];prop=oracle_residual(a,g);new=a+gate*prop;be=float(e089.valid_epe(a,g,v).mean());ne=float(e089.valid_epe(new,g,v).mean());out.append({'pair_id':z['pair_id'],'baseline_aepe':be,'oracle_gate_aepe':ne,'gate_mean':float(gate.mean())})
 stage=f'{kind}_{n}pair_join'
 if kind=='A1': result={'status':'COMPLETE','last_completed_stage':'A1_GT_JOIN','pairs':n,'top1_percent':100*top1/count,'top5_percent':100*top5/count,'entropy_mean':float(np.mean(ents)),'random_top1':100/81,'random_top5':500/81,'pair_rows':out,'prediction_materialized_before_gt_join':True}
 else:
  b=metrics([x['baseline_aepe'] for x in out]);q=metrics([x['oracle_gate_aepe'] for x in out]);result={'status':'COMPLETE','last_completed_stage':'A2_ORACLE_PROPOSAL_GT_JOIN','pairs':n,'baseline':b,'oracle_gate':q,'aepe_relative_reduction':(b['aepe']-q['aepe'])/b['aepe'],'gate_mean':float(np.mean([x['gate_mean'] for x in out])),'pair_rows':out,'prediction_materialized_before_gt_join':True,'oracle_proposal_diagnostic_only':True}
 result['elapsed_seconds']=time.time()-t;card(stage,result)
def reduce():
 a1={n:json.loads((RUN/f'A1_{n}pair_join/RESULT_CARD.json').read_text()) for n in (1,4,16)};a2={n:json.loads((RUN/f'A2_{n}pair_join/RESULT_CARD.json').read_text()) for n in (1,4,16)}
 a1g={1:a1[1]['top1_percent']>=90 and a1[1]['top1_percent']>=2*100/81,4:a1[4]['top1_percent']>=60 and a1[4]['top1_percent']>=2*100/81,16:a1[16]['top1_percent']>=30 and a1[16]['top1_percent']>=2*100/81};a2g={n:a2[n]['aepe_relative_reduction']>=.5 for n in (1,4,16)}
 a1pass=all(a1g.values());a2pass=all(a2g.values());direction='E090C' if a1pass else ('E090B' if a2pass else 'STOP')
 raw=json.loads((E089/'P1_join_r2/RESULT_CARD.json').read_text());a3={'raw_e089':{'candidate_ce_first':raw['candidate_ce_first10_mean'],'candidate_ce_last':raw['candidate_ce_last10_mean'],'gate_mean':raw['gate_activation_mean'],'action_delta':raw['action_path_delta_mean']},'candidate_only':{n:{'ce':json.loads((RUN/f'A1_{n}pair_train/RESULT_CARD.json').read_text())['ce_last'],'top1':a1[n]['top1_percent']} for n in (1,4,16)},'staged_candidate_then_gate':'AUTHORIZED_ONLY_IF_A1_PASS; not run when A1 fails','gate_strength_zero_gradient_diagnosis':'E089 clamp at exactly zero suppresses negative optimizer steps; P0 manually set 0.1 before action test. A2 removes clamped scalar strength.','effective_lr':{'A1':8e-4,'A2':1e-3,'E089_raw':2e-4}}
 result={'status':'COMPLETE','last_completed_stage':'E090A_REPRESENTATION_OPTIMIZATION_DECOUPLED','A1':a1,'A1_gates':a1g,'A1_pass':a1pass,'A2':a2,'A2_gates':a2g,'A2_pass':a2pass,'A3':a3,'root_cause':('candidate representation is learnable; joint optimization suppressed action' if a1pass else ('candidate representation fails but gate can act on oracle proposals' if a2pass else 'both candidate representation and oracle-proposal actionability fail')),'next_direction':direction,'whu_700_test_used':False,'e081_checkpoint_modified':False,'os_modified':False,'lucky_r2_started':False};card('REDUCE',result);print(json.dumps(result,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['a1_train','a1_infer','a2_train','a2_infer','join','reduce'],required=True);p.add_argument('--pairs',type=int,choices=[1,4,16]);p.add_argument('--kind',choices=['A1','A2']);p.add_argument('--steps',type=int,default=1200);a=p.parse_args();started=time.time()
 try:
  if a.mode=='a1_train':a1_train(a.pairs,a.steps)
  elif a.mode=='a1_infer':a1_infer(a.pairs)
  elif a.mode=='a2_train':a2_train(a.pairs,a.steps)
  elif a.mode=='a2_infer':a2_infer(a.pairs)
  elif a.mode=='join':join_arm(a.kind,a.pairs)
  else:reduce()
 except BaseException as e:
  st='OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else ('INTERRUPTED' if isinstance(e,(KeyboardInterrupt,InterruptedError)) else 'IMPLEMENTATION_FAILED');name=f'{a.kind or a.mode}_{a.pairs or ""}_failure';card(name,{'status':st,'last_completed_stage':a.mode,'cause':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started,'e081_checkpoint_modified':False,'os_modified':False});raise
if __name__=='__main__':main()
