#!/usr/bin/env python3
"""E091A 200-group train / 100-group held-out dev validation."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,signal,sys,time,traceback
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(os.environ.get('SPARSEGEO_ROOT','/mnt/sda/zhushihao/VS_Match'));TMP=ROOT/'image_matching_part2_migrated/.codex_tmp';RUN=ROOT/'GDROS/experiments/sparsegeo/runs/E-091A-WHU-200GROUP-HELDOUT-GENERALIZATION';E089=ROOT/'GDROS/experiments/sparsegeo/runs/E-089A-WHU-HIGHRES-CROSSMODAL-RESIDUAL-REFINER';P0=E089/'P0';EXPECTED='4122c27a7a0a1d0cf6488b9207277b0024d2fe0a9f07f6fcd8eec8ef09ca745a'
sys.path.insert(0,str(TMP));import e089a_whu_highres_refiner as e089;import e090d_recurrent_evidence_injection as d90
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def card(s,x):d=RUN/s;dump(d/'RESULT_CARD.json',x);(d/'RESULT_CARD.md').write_text(f'# E091A {s}\n\nStatus: `{x["status"]}`\n\nLast stage: `{x.get("last_completed_stage")}`\n')
def split():return json.loads((RUN/'SPLIT_MANIFEST.json').read_text())
def prepare():
 src=json.loads((P0/'SPLIT_MANIFEST.json').read_text());groups=sorted({r['source_group'] for r in src['train_pairs']},key=lambda g:hashlib.sha256(f'E091A-train-v1:{g}'.encode()).hexdigest())[:200];gs=set(groups);train=[r for r in src['train_pairs'] if r['source_group'] in gs];dev=src['dev_pairs'];assert len(train)==1600 and len(dev)==800 and not gs&{r['source_group'] for r in dev};cache=json.loads((E089/'P0_cache/RESULT_CARD.json').read_text());assert cache['pairs']==5600 and cache['checkpoint_sha256']==EXPECTED and sha(Path(cache['manifest']))==cache['manifest_sha256'];x={'status':'FROZEN','train_pairs':train,'dev_pairs':dev,'train_groups':groups,'dev_groups':sorted({r['source_group'] for r in dev}),'group_overlap':0,'selection':'lowest 200 SHA256(E091A-train-v1:<group>) from E089 600 train groups','anchor_cache_manifest':cache['manifest'],'anchor_cache_manifest_sha256':cache['manifest_sha256'],'whu_700_test_sealed':True};dump(RUN/'SPLIT_MANIFEST.json',x);card('PREPARE',{'status':'PASS','last_completed_stage':'200_100_GROUP_SPLIT_AND_CACHE_AUDIT','train_pairs':1600,'dev_pairs':800,'train_groups':200,'dev_groups':100,'group_overlap':0,'base_checkpoint_sha256':EXPECTED,'anchor_cache_verified':True,'test_700_sealed':True})
def get_batch(ds,row,dev,gt=True):
 if gt:_,o,s,g,v=ds[row['pair_index']];g,v=g[None].to(dev),v[None].to(dev)
 else:o,s=e089.read_images(ds,row['pair_index']);g=v=None
 a=torch.load(P0/'anchor_cache'/f'{row["pair_id"]}.pt',map_location='cpu');return o[None].to(dev),s[None].to(dev),a['forward'][None].float().to(dev),g,v
def epoch_key(phase,epoch):return f'{phase}_epoch{epoch}'
def previous(phase,epoch):
 if phase=='candidate':return None if epoch==1 else RUN/f'candidate_epoch{epoch-1}_train/checkpoint.pt'
 return RUN/f'candidate_epoch4_train/checkpoint.pt' if epoch==1 else RUN/f'recurrent_epoch{epoch-1}_train/checkpoint.pt'
def train(phase,epoch):
 t=time.time();dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.manual_seed(9100+(0 if phase=='candidate' else 100)+epoch);m=d90.load(dev);prev=previous(phase,epoch)
 if prev:m.load_state_dict(torch.load(prev,map_location=dev)['model'])
 train_rows=split()['train_pairs'];ds=e089.training_dataset();presentations=len(train_rows);curve=[]
 if phase=='candidate':params=list(m.local.opt_stem.parameters())+list(m.local.sar_stem.parameters())+list(m.local.shared_projection.parameters())+list(m.local.score_head.parameters())+[m.local.log_temperature,m.dustbin];lr=4e-4
 else:
  if epoch==1:
   with torch.no_grad():m.adapter[-1].weight.fill_(1e-5)
  params=[p for p in m.parameters() if p.requires_grad];lr=1e-4
 opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-5);start=time.time()
 for step,row in enumerate(train_rows,1):
  o,s,a,g,v=get_batch(ds,row,dev);anchor=a
  if phase=='candidate':src,tgt=m.local.descriptors(o,s);z=m.local.cost_volume(src,tgt,anchor);lab,vh,inside=d90.candidate_loss(z,anchor,g,v);dust=m.dustbin.expand(z.shape[0],1,*z.shape[-2:]);ce=F.cross_entropy(torch.cat((z,dust),1),lab,reduction='none')[vh].mean();loss=ce;out=anchor
  else:outs,x=m(o,s,enabled=True,anchor_override=anchor);out=d90.final_tensor(outs);lab,vh,inside=d90.candidate_loss(x['logits'],anchor,g,v);dust=m.dustbin.expand(x['logits'].shape[0],1,*x['logits'].shape[-2:]);ce=F.cross_entropy(torch.cat((x['logits'],dust),1),lab,reduction='none')[vh].mean();epe=torch.sqrt(e089.valid_epe(out,g,v).square()+.01).mean();loss=.25*ce+epe
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,5);opt.step()
  if step==1 or step%100==0:curve.append({'presentation':step,'loss':float(loss),'ce':float(ce),'aepe':float(e089.valid_epe(out,g,v).mean()),'lr':opt.param_groups[0]['lr']});print(json.dumps(curve[-1]),flush=True)
 d=RUN/f'{epoch_key(phase,epoch)}_train';p=d/'checkpoint.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save({'model':m.state_dict(),'phase':phase,'epoch':epoch,'fresh_base':phase=='candidate' and epoch==1,'base_sha':EXPECTED},p);dump(d/'CURVE.json',curve);card(f'{epoch_key(phase,epoch)}_train',{'status':'COMPLETE','last_completed_stage':'FULL_EPOCH_TRAIN','phase':phase,'epoch':epoch,'pair_presentations':presentations,'optimizer_steps':presentations,'effective_batch':1,'gradient_accumulation':1,'lr':lr,'elapsed_seconds':time.time()-t,'training_seconds':time.time()-start,'checkpoint':str(p),'checkpoint_sha256':sha(p),'initialization':'fresh E081A + zero E090D' if phase=='candidate' and epoch==1 else str(prev),'base_checkpoint_sha256':EXPECTED,'test_700_used':False})
def load_epoch(phase,epoch,dev):
 c=json.loads((RUN/f'{epoch_key(phase,epoch)}_train/RESULT_CARD.json').read_text());m=d90.load(dev);m.load_state_dict(torch.load(c['checkpoint'],map_location=dev)['model']);m.eval();return m
def infer(phase,epoch,shard,shards=4):
 t=time.time();dev=torch.device('cuda:0');torch.cuda.set_device(dev);m=load_epoch(phase,epoch,dev);rows=[r for i,r in enumerate(split()['dev_pairs']) if i%shards==shard];ds=e089.no_gt_dataset('training');od=RUN/f'{epoch_key(phase,epoch)}_infer/shard{shard}';od.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for i,row in enumerate(rows):
   o,s,a,_,_=get_batch(ds,row,dev,False)
   if phase=='candidate':
    src,tgt=m.local.descriptors(o,s);logits=m.local.cost_volume(src,tgt,a);_ev,stats=m.stats(logits);main=off=zero=shuffle=a;sequence_delta=[0.0]*32;x={'logits':logits,'stats':stats}
   else:
    main,x=m(o,s,enabled=True,anchor_override=a);off,_=m(o,s,enabled=False,anchor_override=a);zero,_=m(o,s,enabled=True,evidence_mode='zero',anchor_override=a);shuffle,_=m(o,s,enabled=True,evidence_mode='shuffle',anchor_override=a);sequence_delta=[float(d90.final_tensor(q).abs().mean()) for q in x['deltas']]
   torch.save({'pair_index':row['pair_index'],'pair_id':row['pair_id'],'source_group':row['source_group'],'baseline':a[0].half().cpu(),'main':d90.final_tensor(main)[0].half().cpu(),'off':d90.final_tensor(off)[0].half().cpu(),'zero':d90.final_tensor(zero)[0].half().cpu(),'shuffle':d90.final_tensor(shuffle)[0].half().cpu(),'logits':x['logits'][0].half().cpu(),'entropy':x['stats']['entropy'][0].half().cpu(),'sequence_delta':sequence_delta,'uses_gt':False},od/f'{row["pair_id"]}.pt')
   if i%50==0:print(json.dumps({'shard':shard,'done':i+1,'total':len(rows)}),flush=True)
 fs=sorted(od.glob('*.pt'));assert len(fs)==len(rows);man=RUN/f'{epoch_key(phase,epoch)}_infer/SHARD{shard}.jsonl';man.write_text(''.join(json.dumps({'path':str(p),'sha':sha(p),'uses_gt':False})+'\n' for p in fs));card(f'{epoch_key(phase,epoch)}_infer_shard{shard}',{'status':'COMPLETE','last_completed_stage':'DEV_PREDICTIONS_BEFORE_GT','phase':phase,'epoch':epoch,'shard':shard,'pairs':len(rows),'manifest':str(man),'manifest_sha256':sha(man),'prediction_materialized_before_gt_join':True,'elapsed_seconds':time.time()-t})
def metric(vals):
 a=np.asarray(vals);out={'pairs':len(a),'aepe':float(a.mean()),'fail_ge5':int((a>=5).sum())};out.update({f'cmr{k}':float(100*(a<k).mean()) for k in (1,2,5,10,16)});return out
def bootstrap(base,new,seed=911):
 b=np.asarray(base);n=np.asarray(new);rng=np.random.default_rng(seed);ix=rng.integers(0,len(b),(2000,len(b)));da=(b[ix]-n[ix]).mean(1);dc=(100*((n[ix]<1).mean(1)-(b[ix]<1).mean(1)));return {'aepe_improvement_95ci':[float(np.quantile(da,.025)),float(np.quantile(da,.975))],'cmr1_gain_pp_95ci':[float(np.quantile(dc,.025)),float(np.quantile(dc,.975))]}
def join(phase,epoch):
 t=time.time();recs=[]
 for shard in range(4):
  c=json.loads((RUN/f'{epoch_key(phase,epoch)}_infer_shard{shard}/RESULT_CARD.json').read_text());m=Path(c['manifest']);assert sha(m)==c['manifest_sha256'];recs += [json.loads(x) for x in m.read_text().splitlines()]
 assert len(recs)==800;ds=e089.training_dataset();rows=[];cand1=cand5=candn=0;ents=[]
 for rec in recs:
  z=torch.load(rec['path']);_,o,s,g,v=ds[z['pair_index']];g,v=g[None],v[None];row={'pair_id':z['pair_id'],'source_group':z['source_group']}
  for name in ('baseline','main','off','zero','shuffle'):row[f'{name}_aepe']=float(e089.valid_epe(z[name][None].float(),g,v).mean())
  row['off_max_abs']=float((z['off'].float()-z['baseline'].float()).abs().max());row['sequence_delta']=z['sequence_delta'];lab,vh,inside=d90.candidate_loss(z['logits'][None].float(),z['baseline'][None].float(),g,v);rank=z['logits'].float().argsort(0,descending=True);cand1+=int((rank[0][inside[0]]==lab[0][inside[0]]).sum());cand5+=int((rank[:5,inside[0]]==lab[0][inside[0]][None]).any(0).sum());candn+=int(inside.sum());ents.append(float(z['entropy'].float().mean()));rows.append(row)
 methods={name:metric([r[f'{name}_aepe'] for r in rows]) for name in ('baseline','main','off','zero','shuffle')};ci=bootstrap([r['baseline_aepe'] for r in rows],[r['main_aepe'] for r in rows],9100+epoch+(0 if phase=='candidate' else 10));groups={}
 for r in rows:groups.setdefault(str(r['source_group']),[]).append(r['baseline_aepe']-r['main_aepe'])
 group_summary={k:float(np.mean(v)) for k,v in groups.items()};main_delta=np.asarray([r['baseline_aepe']-r['main_aepe'] for r in rows]);zero_delta=np.asarray([r['zero_aepe']-r['main_aepe'] for r in rows]);shuffle_delta=np.asarray([r['shuffle_aepe']-r['main_aepe'] for r in rows]);seq=np.mean([r['sequence_delta'] for r in rows],axis=0).tolist()
 result={'status':'COMPLETE','last_completed_stage':'FULL_800_DEV_GT_JOIN','phase':phase,'epoch':epoch,'methods':methods,'bootstrap_95ci':ci,'candidate_top1':100*cand1/max(candn,1),'candidate_top5':100*cand5/max(candn,1),'candidate_entropy_mean':float(np.mean(ents)),'sequence_action_delta_mean':seq,'adapter_off_max_abs':max(r['off_max_abs'] for r in rows),'main_minus_zero_aepe_improvement':float(zero_delta.mean()),'main_minus_shuffle_aepe_improvement':float(shuffle_delta.mean()),'main_improved_pairs':int((main_delta>0).sum()),'main_degraded_pairs':int((main_delta<0).sum()),'failure_buckets':{k:int(sum(e089_bucket(r['main_aepe'])==k for r in rows)) for k in ('<1','[1,2)','[2,5)','[5,10)','[10,16)','>=16')},'source_group_mean_improvement':group_summary,'pair_rows':rows,'prediction_materialized_before_gt_join':True,'test_700_used':False,'elapsed_seconds':time.time()-t};card(f'{epoch_key(phase,epoch)}_join',result)
def e089_bucket(v):
 if v<1:return '<1'
 if v<2:return '[1,2)'
 if v<5:return '[2,5)'
 if v<10:return '[5,10)'
 if v<16:return '[10,16)'
 return '>=16'
def select():
 allrows=[]
 for phase,epochs in (('candidate',range(1,5)),('recurrent',range(1,4))):
  for epoch in epochs:
   x=json.loads((RUN/f'{epoch_key(phase,epoch)}_join/RESULT_CARD.json').read_text());m=x['methods']['main'];allrows.append({'phase':phase,'epoch':epoch,'metrics':m,'card':str(RUN/f'{epoch_key(phase,epoch)}_join/RESULT_CARD.json')})
 winner=max(allrows,key=lambda r:(r['metrics']['cmr1'],-r['metrics']['aepe'],r['metrics']['cmr2'],r['metrics']['cmr5']));x=json.loads(Path(winner['card']).read_text());b=x['methods']['baseline'];m=x['methods']['main'];z=x['methods']['zero'];s=x['methods']['shuffle'];gate={'cmr1_gain_ge3':m['cmr1']-b['cmr1']>=3,'aepe_improvement_ge_0_05':b['aepe']-m['aepe']>=.05,'cmr2_drop_le_0_5':m['cmr2']>=b['cmr2']-.5,'cmr5_drop_le_0_5':m['cmr5']>=b['cmr5']-.5,'failures_not_increased':m['fail_ge5']<=b['fail_ge5'],'off_identity':x['adapter_off_max_abs']==0,'main_beats_zero_significantly':x['main_minus_zero_aepe_improvement']>0,'main_beats_shuffle_significantly':x['main_minus_shuffle_aepe_improvement']>0,'bootstrap_aepe_ci_positive':x['bootstrap_95ci']['aepe_improvement_95ci'][0]>0}
 result={'status':'PASS_RETURN_TO_METHOD' if all(gate.values()) else 'STOP_HELDOUT_GATE_FAILED','last_completed_stage':'PREDECLARED_DEV_SELECTION_AND_GATE','selection_order':'CMR@1 -> AEPE -> CMR@2 -> CMR@5','winner':winner,'winner_dev_card':x,'gates':gate,'all_epochs':allrows,'train_presentations':{'candidate':6400,'recurrent':4800},'train_groups':200,'dev_groups':100,'whu_700_test_used':False,'dev_full_test_expansion_started':False,'root_cause':None if all(gate.values()) else ('generalization/evidence/optimization')};card('REDUCE',result)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','train','infer','join','select'],required=True);p.add_argument('--phase',choices=['candidate','recurrent']);p.add_argument('--epoch',type=int);p.add_argument('--shard',type=int);a=p.parse_args();start=time.time()
 try:
  if a.mode=='prepare':prepare()
  elif a.mode=='train':train(a.phase,a.epoch)
  elif a.mode=='infer':infer(a.phase,a.epoch,a.shard)
  elif a.mode=='join':join(a.phase,a.epoch)
  else:select()
 except BaseException as e:
  name=f'{a.mode}_{a.phase}_{a.epoch}_{a.shard}_failure';card(name,{'status':'OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else ('INTERRUPTED' if isinstance(e,(KeyboardInterrupt,InterruptedError)) else 'IMPLEMENTATION_FAILED'),'last_completed_stage':a.mode,'cause':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start,'base_checkpoint_modified':False,'test_700_used':False});raise
if __name__=='__main__':main()
