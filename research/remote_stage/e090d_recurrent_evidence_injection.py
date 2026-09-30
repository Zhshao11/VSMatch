#!/usr/bin/env python3
"""E090D structured local evidence injection into late recurrent updates."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np,torch
import torch.nn as nn
import torch.nn.functional as F
ROOT=Path(os.environ.get('SPARSEGEO_ROOT','/mnt/sda/zhushihao/VS_Match'));TMP=ROOT/'image_matching_part2_migrated/.codex_tmp';RUN=ROOT/'GDROS/experiments/sparsegeo/runs/E-090D-WHU-RECURRENT-EVIDENCE-INJECTION';E089=ROOT/'GDROS/experiments/sparsegeo/runs/E-089A-WHU-HIGHRES-CROSSMODAL-RESIDUAL-REFINER';P0=E089/'P0';EXPECTED='4122c27a7a0a1d0cf6488b9207277b0024d2fe0a9f07f6fcd8eec8ef09ca745a'
sys.path.insert(0,str(TMP));import e089a_whu_highres_refiner as e089
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def card(s,x):d=RUN/s;dump(d/'RESULT_CARD.json',x);(d/'RESULT_CARD.md').write_text(f'# E090D {s}\n\nStatus: `{x["status"]}`\n\nLast stage: `{x.get("last_completed_stage")}`\n')
def rows(n):return json.loads((E089/'P1_train/PAIR_MANIFEST.json').read_text())['pairs'][:n]
def batch(ds,row,dev,gt=True):
 if gt:_,o,s,g,v=ds[row['pair_index']];g,v=g[None].to(dev),v[None].to(dev)
 else:o,s=e089.read_images(ds,row['pair_index']);g=v=None
 a=torch.load(P0/'anchor_cache'/f'{row["pair_id"]}.pt',map_location='cpu');return o[None].to(dev),s[None].to(dev),a['forward'][None].float().to(dev),g,v
class EvidenceStats(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Conv2d(10,24,3,2,1),nn.GELU(),nn.Conv2d(24,24,3,2,1),nn.GELU())
 def forward(self,logits):
  p=torch.softmax(logits/.07,1);top=torch.topk(p,2,1);off=torch.tensor(e089.OFFSETS,device=p.device,dtype=p.dtype);ex=(p[:,None]*off.T[None,:,:,None,None]).sum(2);center=off.T[None,:,:,None,None]-ex[:,:,None];sec=(p[:,None]*center.square()).sum(2);ent=-(p*p.clamp_min(1e-9).log()).sum(1,keepdim=True)/math.log(81);idx=top.indices[:,0];mode=off[idx].permute(0,3,1,2);raw=torch.cat((ex,sec,ent,top.values[:,:1]-top.values[:,1:2],mode,top.values[:,:1],logits.mean(1,keepdim=True)),1);return self.encoder(raw),{'expected':ex,'second':sec,'entropy':ent,'margin':top.values[:,:1]-top.values[:,1:2],'top_modes':mode}
class E090D(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.local=e089.HighResolutionResidualRefiner();self.stats=EvidenceStats();self.dustbin=nn.Parameter(torch.zeros(()));self.late=copy.deepcopy(base.backbone.update_block);self.adapter=nn.Sequential(nn.Conv2d(24+2+2,32,3,1,1),nn.GELU(),nn.Conv2d(32,2,3,1,1));nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
  for p in self.base.parameters():p.requires_grad_(False)
  for p in self.local.gate_net.parameters():p.requires_grad_(False)
  self.local.gate_strength.requires_grad_(False)
  for n,p in self.late.named_parameters():p.requires_grad_('encoder' in n or 'flow_head' in n)
 def forward(self,o,s,iters=32,enabled=True,evidence_mode='main',anchor_override=None):
  from core.corr import CorrBlock
  b=self.base.backbone;i1=2*(o/255)-1;i2=2*(s/255)-1;ctx=b.context_encoder(i1);f1,f2=b.featureNet(i1,i2,{},ctx);corrfn=CorrBlock(f1.float(),f2.float(),radius=b.corr_radius);c=b.context_encoder(i1);net,inp=torch.split(c,[b.hidden_dim,b.context_dim],1);net,inp=torch.tanh(net),torch.relu(inp);c0,c1=b.initialize_flow(i1)
  # Frozen anchor for high-resolution evidence; no GT.
  with torch.no_grad(): anchor=e089.frozen_anchor(self.base,o,s) if anchor_override is None else anchor_override
  src,tgt=self.local.descriptors(o,s);logits=self.local.cost_volume(src,tgt,anchor);ev,diag=self.stats(logits)
  if evidence_mode=='zero':ev=torch.zeros_like(ev)
  elif evidence_mode=='shuffle':ev=torch.flip(ev,dims=(2,3))
  elif evidence_mode!='main':raise ValueError(evidence_mode)
  if not enabled:
   zero=torch.zeros_like(c1);return [anchor for _ in range(iters)],{'logits':logits,'evidence':ev,'stats':diag,'deltas':[zero for _ in range(iters)],'anchor':anchor}
  outs=[];deltas=[]
  ref_net=ref_c1=None
  for k in range(iters):
   before=c1.detach();corr=corrfn(before);flow=before-c0;base_net,mask,base_delta=b.update_block(net,inp,corr,flow)
   if k>=iters-4:
    if k==iters-4:
     ref_net=base_net.clone();ref_c1=before+base_delta;ref_mask=mask;ref_delta=base_delta
    else:
     ref_before=ref_c1.detach();ref_corr=corrfn(ref_before);ref_net,ref_mask,ref_delta=b.update_block(ref_net,inp,ref_corr,ref_before-c0);ref_c1=ref_before+ref_delta
    late_net,late_mask,late_delta=self.late(net,inp,corr,flow);e=F.interpolate(ev,flow.shape[-2:],mode='bilinear',align_corners=True);cor=self.adapter(torch.cat((e,flow,late_delta-base_delta),1));delta=base_delta+cor;net=base_net
    c1=before+delta;correction=b.upsample_flow(c1-ref_c1,ref_mask);outs.append(anchor+correction);deltas.append(delta-ref_delta)
   else:
    delta=base_delta;net=base_net;c1=before+delta;outs.append(anchor);deltas.append(torch.zeros_like(delta))
  return outs,{'logits':logits,'evidence':ev,'stats':diag,'deltas':deltas,'anchor':anchor}
def load(dev):
 _,base=e089.load_e081(dev);return E090D(base).to(dev)
def design():
 x={'status':'COMPLETE','last_completed_stage':'DESIGN_FROZEN','path':'frozen raw CorrBlock -> frozen base update for all 32 iterations; structured 9x9 evidence -> evidence encoder -> zero residual adapter only in iterations 29-32; trainable late update copy supplies restricted motion/update delta features','evidence_tensor':'stride2 logits [B,81,H/2,W/2] -> expected residual(2), second moment(2), entropy(1), margin(1), top1 mode(2), top1 mass(1), mean logit(1) -> stride8 [B,24,H/8,W/8]','identity':'adapter final conv zero; disabled path uses only frozen base update','trainable':'OPT/SAR stems, shared projection, score head, evidence encoder, adapter, late update encoder+flow_head only','frozen':'E081 feature/context networks, raw CorrBlock, base update block, M4 refiner, LSR, remaining late update parameters','candidate_loss':'r4 labels only plus explicit dustbin for outside-window tokens','test_700_sealed':True};card('D0_design',x)
def g0():
 t=time.time();dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.manual_seed(990)
 m=load(dev);ds=e089.training_dataset();o,s,a,g,v=batch(ds,rows(1)[0],dev);torch.cuda.reset_peak_memory_stats()
 off,_=m(o,s,enabled=False);on,x=m(o,s,enabled=True);identity=float((off[-1]-on[-1]).abs().max())
 loss=F.cross_entropy(x['logits'],torch.zeros_like(x['logits'][:,0],dtype=torch.long))+on[-1].square().mean();loss.backward()
 auth={n:float(p.grad.abs().sum()) if p.grad is not None else 0 for n,p in m.named_parameters() if p.requires_grad}
 frozen_bad=[n for n,p in m.named_parameters() if not p.requires_grad and p.grad is not None]
 blocked_upstream=sum(value for name,value in auth.items() if name.startswith(('stats.','late.','adapter.0')))
 with torch.no_grad():m.adapter[-1].weight.fill_(1e-4)
 m.zero_grad(set_to_none=True);changed,changed_aux=m(o,s,enabled=True);synthetic=float((changed[-1]-off[-1]).abs().mean());changed[-1].square().mean().backward()
 auth2={n:float(p.grad.abs().sum()) if p.grad is not None else 0 for n,p in m.named_parameters() if p.requires_grad}
 groups={'descriptor':('local.opt_stem','local.sar_stem','local.shared_projection'),'evidence':('stats.',),'late_update':('late.',),'adapter':('adapter.',)}
 group_nonzero={group:any(value>0 and np.isfinite(value) for name,value in auth2.items() if name.startswith(prefixes)) for group,prefixes in groups.items()}
 checks={'identity_parity_max_abs_le_1e-6':identity<=1e-6,'adapter_gradient_nonzero':sum(value for name,value in auth.items() if name.startswith('adapter.'))>0,'stats_late_blocked_at_exact_zero':blocked_upstream==0,'all_authorized_groups_nonzero_after_epsilon_adapter':all(group_nonzero.values()),'frozen_gradient_absent':not frozen_bad,'synthetic_evidence_changes_delta':synthetic>0,'finite':all(torch.isfinite(z).all() for z in on),'shape':list(x['evidence'].shape)==[1,24,64,64]}
 res={'status':'PASS' if all(checks.values()) else 'FAIL','last_completed_stage':'D0_G0','checks':checks,'identity_max_abs':identity,'synthetic_action_delta':synthetic,'zero_init_gradients':auth,'epsilon_adapter_group_nonzero':group_nonzero,'frozen_bad':frozen_bad,'peak_memory_mb':int(torch.cuda.max_memory_allocated()//2**20),'elapsed_seconds':time.time()-t,'next_stage':'D1_ALLOWED' if all(checks.values()) else 'STOP'}
 card('D0_g0',res);assert res['status']=='PASS'
def candidate_loss(logits,anchor,truth,valid):
 r=F.interpolate(truth-anchor,logits.shape[-2:],mode='bilinear',align_corners=True);inside=(r[:,0].abs()<=4)&(r[:,1].abs()<=4);off=torch.tensor(e089.OFFSETS,device=logits.device);lab=torch.linalg.vector_norm(r.permute(0,2,3,1)[...,None,:]-off[None,None,None],dim=-1).argmin(-1);lab=torch.where(inside,lab,torch.full_like(lab,81));vh=F.interpolate(valid[:,None].float(),lab.shape[-2:],mode='nearest')[:,0].bool();return lab,vh,inside&vh
def final_tensor(value):
 while isinstance(value,(list,tuple)):value=value[-1]
 return value
def d1_train(n,candidate_steps=500,recurrent_steps=400):
 t=time.time();dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.manual_seed(1000+n);m=load(dev);ds=e089.training_dataset();local_params=list(m.local.opt_stem.parameters())+list(m.local.sar_stem.parameters())+list(m.local.shared_projection.parameters())+list(m.local.score_head.parameters())+[m.local.log_temperature,m.dustbin];opt=torch.optim.AdamW(local_params,lr=8e-4);curve=[]
 for step in range(1,candidate_steps+recurrent_steps+1):
  row=rows(n)[(step-1)%n];o,s,a,g,v=batch(ds,row,dev);anchor=a
  if step<=candidate_steps:
   src,tgt=m.local.descriptors(o,s);z=m.local.cost_volume(src,tgt,anchor);lab,vh,inside=candidate_loss(z,anchor,g,v);dust=m.dustbin.expand(z.shape[0],1,*z.shape[-2:]);ce=F.cross_entropy(torch.cat((z,dust),1),lab,reduction='none')[vh].mean();loss=ce;out=anchor;aux=None
  else:
   if step==candidate_steps+1:
    with torch.no_grad():m.adapter[-1].weight.fill_(1e-5)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-4)
   out,aux=m(o,s,enabled=True);lab,vh,inside=candidate_loss(aux['logits'],anchor,g,v);dust=m.dustbin.expand(aux['logits'].shape[0],1,*aux['logits'].shape[-2:]);ce=F.cross_entropy(torch.cat((aux['logits'],dust),1),lab,reduction='none')[vh].mean();epe=torch.sqrt(e089.valid_epe(final_tensor(out),g,v).square()+.01).mean();loss=.25*ce+epe
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],5);opt.step()
  final_out=out if step<=candidate_steps else final_tensor(out)
  if step==1 or step%25==0:curve.append({'step':step,'phase':'candidate' if step<=candidate_steps else 'recurrent','loss':float(loss),'ce':float(ce),'aepe':float(e089.valid_epe(final_out,g,v).mean()),'action_delta':float((final_out-anchor).abs().mean()),'adapter_norm':float(sum(p.detach().square().sum() for p in m.adapter.parameters()).sqrt())});print(json.dumps(curve[-1]),flush=True)
 d=RUN/f'D1_{n}pair_train';d.mkdir(parents=True,exist_ok=True);p=d/f'E090D_{n}.pt';torch.save({'model':m.state_dict(),'pairs':n},p);dump(d/'CURVE.json',curve);card(f'D1_{n}pair_train',{'status':'COMPLETE','last_completed_stage':'D1_PHASED_TRAIN','pairs':n,'candidate_steps':candidate_steps,'recurrent_steps':recurrent_steps,'checkpoint':str(p),'checkpoint_sha256':sha(p),'curve_tail':curve[-5:],'test_700_used':False})
def d1_infer(n):
 dev=torch.device('cuda:0');torch.cuda.set_device(dev);tr=json.loads((RUN/f'D1_{n}pair_train/RESULT_CARD.json').read_text());m=load(dev);m.load_state_dict(torch.load(tr['checkpoint'],map_location=dev)['model']);m.eval();ds=e089.no_gt_dataset('training');d=RUN/f'D1_{n}pair_infer/predictions';d.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for row in rows(n):
   o,s,a,_,_=batch(ds,row,dev,False);new,x=m(o,s,enabled=True);off,_=m(o,s,enabled=False);torch.save({'pair_index':row['pair_index'],'pair_id':row['pair_id'],'baseline':a[0].half().cpu(),'new':final_tensor(new)[0].half().cpu(),'off':final_tensor(off)[0].half().cpu(),'logits':x['logits'][0].half().cpu(),'sequence_delta':[float(final_tensor(q).abs().mean()) for q in x['deltas']],'adapter_norm':float(sum(p.square().sum() for p in m.adapter.parameters()).sqrt()),'uses_gt':False},d/f'{row["pair_id"]}.pt')
 fs=sorted(d.glob('*.pt'));man=RUN/f'D1_{n}pair_infer/PREDICTIONS.jsonl';man.write_text(''.join(json.dumps({'path':str(p),'sha':sha(p),'uses_gt':False})+'\n' for p in fs));card(f'D1_{n}pair_infer',{'status':'COMPLETE','last_completed_stage':'PREDICTIONS_BEFORE_GT','pairs':n,'manifest':str(man),'manifest_sha256':sha(man),'prediction_materialized_before_gt_join':True})
def met(a):
 x=np.asarray(a);return {'aepe':float(x.mean()),'cmr1':float(100*(x<1).mean()),'cmr2':float(100*(x<2).mean()),'cmr5':float(100*(x<5).mean())}
def d1_join(n):
 inf=json.loads((RUN/f'D1_{n}pair_infer/RESULT_CARD.json').read_text());recs=[json.loads(x) for x in Path(inf['manifest']).read_text().splitlines()];ds=e089.training_dataset();br=[];nr=[];offmax=0;top1=top5=count=0;seq=[]
 for rec in recs:
  z=torch.load(rec['path']);_,o,s,g,v=ds[z['pair_index']];g,v=g[None],v[None];b=z['baseline'][None].float();new=z['new'][None].float();br.append(float(e089.valid_epe(b,g,v).mean()));nr.append(float(e089.valid_epe(new,g,v).mean()));offmax=max(offmax,float((z['off'].float()-z['baseline'].float()).abs().max()));lab,vh,inside=candidate_loss(z['logits'][None].float(),b,g,v);rank=z['logits'].float().argsort(0,descending=True);top1+=int((rank[0][inside[0]]==lab[0][inside[0]]).sum());top5+=int((rank[:5,inside[0]]==lab[0][inside[0]][None]).any(0).sum());count+=int(inside.sum());seq.append(z['sequence_delta'])
 B,N=met(br),met(nr);red=(B['aepe']-N['aepe'])/B['aepe'];gates={1:red>=.7 or N['cmr1']-B['cmr1']>=50,4:red>=.5 or N['cmr1']-B['cmr1']>=30,8:red>=.3 and N['cmr1']-B['cmr1']>=15};passed=gates[n] and N['cmr2']>=B['cmr2'] and N['cmr5']>=B['cmr5'] and offmax==0
 card(f'D1_{n}pair_join',{'status':'PASS' if passed else 'FAIL','last_completed_stage':'D1_GT_JOIN','pairs':n,'baseline':B,'new':N,'aepe_relative_reduction':red,'cmr1_gain_pp':N['cmr1']-B['cmr1'],'candidate_top1':100*top1/count,'candidate_top5':100*top5/count,'sequence_action_delta_mean':np.mean(seq,axis=0).tolist(),'adapter_off_max_abs':offmax,'gate':passed,'prediction_materialized_before_gt_join':True})
def d1_reduce():
 x={n:json.loads((RUN/f'D1_{n}pair_join/RESULT_CARD.json').read_text()) for n in (1,4,8)};p=all(v['gate'] for v in x.values());card('D1_REDUCE',{'status':'PASS_D1_RETURN_TO_METHOD' if p else 'STOP_D1_FAILED','last_completed_stage':'PROGRESSIVE_ACTIONABILITY_REDUCED','arms':x,'dev_authorized':False,'test_700_used':False,'root_cause':None if p else ('evidence_unusable_or_recurrent_optimization_failed'),'next_stage':'METHOD_DECISION_ONLY' if p else 'STOP'})
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['design','g0','train','infer','join','reduce']);p.add_argument('--pairs',type=int,choices=[1,4,8]);a=p.parse_args()
 try:
  if a.mode=='design':design()
  elif a.mode=='g0':g0()
  elif a.mode=='train':d1_train(a.pairs)
  elif a.mode=='infer':d1_infer(a.pairs)
  elif a.mode=='join':d1_join(a.pairs)
  else:d1_reduce()
 except BaseException as e:card(f'D0_{a.mode}_failure',{'status':'OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else 'IMPLEMENTATION_FAILED','last_completed_stage':a.mode,'cause':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
