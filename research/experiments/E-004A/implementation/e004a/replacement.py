import torch

from experiments.sparsegeo.e004a.anchor_heads import POST_FEATURE_DIM
from experiments.sparsegeo.e004a.proposals import build_region_proposals


def build_post_features(pre_logits, candidate_scores, target_xy, coarse_xy, temperature=0.1):
    return build_post_features_subset(pre_logits, candidate_scores, target_xy, coarse_xy, temperature)


def build_post_features_subset(pre_logits, candidate_scores, target_xy, coarse_xy, temperature=0.1):
    count = pre_logits.numel()
    pre_logits = pre_logits.to(torch.float32).reshape(count)
    scores = candidate_scores.to(pre_logits.device, torch.float32).reshape(count, 8)
    target_xy = target_xy.to(pre_logits.device, torch.float32).reshape(count, 8, 2)
    probability = torch.softmax(scores / temperature, dim=-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    top4_mass = probability[:, :4].sum(dim=-1)
    fine_top1 = target_xy[:, 0]
    linf = (fine_top1 - coarse_xy.to(pre_logits.device, torch.float32)).abs().amax(dim=-1)
    features = torch.cat((
        pre_logits[:, None], scores,
        (scores[:, 0] - scores[:, 1])[:, None], entropy[:, None], top4_mass[:, None],
        fine_top1 / 63.0, linf[:, None],
    ), dim=-1)
    if features.shape != (count, POST_FEATURE_DIM):
        raise RuntimeError(f"unexpected post feature shape {features.shape}")
    return features


def plan_reserve_attempts(primary, reserves, pre_probability, primary_post_probability, max_attempts=64):
    proposals=build_region_proposals(); primary=primary.to(torch.long); reserves=reserves.to(torch.long)
    bad=[int(primary[i]) for i in range(128) if float(primary_post_probability[i])<.5]
    bad.sort(key=lambda i:(float(primary_post_probability[(primary==i).nonzero()[0]]),int(proposals.region_id[i]),i))
    used=set(primary.tolist()); attempts=[]
    for old in bad:
        if len(attempts)>=max_attempts: break
        region=int(proposals.region_id[old]); choices=[int(i) for i in reserves if int(proposals.region_id[i])==region and int(i) not in used]
        if not choices: continue
        new=max(choices,key=lambda i:(float(pre_probability[i]),-i)); used.add(new); attempts.append((old,new))
    return attempts


def finalize_planned_replacements(primary, attempts, primary_post_probability, reserve_post_probability):
    final=primary.to(torch.long).tolist(); position={v:i for i,v in enumerate(final)}; trace=[]
    for j,(old,new) in enumerate(attempts):
        old_post=float(primary_post_probability[position[old]]); new_post=float(reserve_post_probability[j]); accepted=new_post>=old_post+.05
        if accepted: final[position[old]]=new
        trace.append({"primary":old,"reserve":new,"region":old//4,"accepted":accepted,"primary_post":old_post,"reserve_post":new_post})
    result=torch.tensor(final,dtype=torch.long)
    if result.numel()!=128 or torch.unique(result).numel()!=128: raise RuntimeError("invalid final K128")
    return result,trace


def replace_same_region_once(primary, reserves, pre_probability, post_probability, max_attempts=64):
    proposals = build_region_proposals()
    primary = primary.to(torch.long).tolist()
    reserves = reserves.to(torch.long).tolist()
    used = set(primary)
    reserve_by_region = {region: [] for region in range(64)}
    for index in reserves:
        reserve_by_region[int(proposals.region_id[index])].append(index)
    bad = [i for i in primary if float(post_probability[i]) < 0.5]
    bad.sort(key=lambda i: (float(post_probability[i]), int(proposals.region_id[i]), i))
    trace, attempted = [], 0
    final = list(primary)
    for old in bad:
        if attempted >= max_attempts:
            break
        region = int(proposals.region_id[old])
        candidates = [i for i in reserve_by_region[region] if i not in used]
        if not candidates:
            continue
        new = max(candidates, key=lambda i: (float(pre_probability[i]), -i))
        attempted += 1; used.add(new)
        accepted = float(post_probability[new]) >= float(post_probability[old]) + 0.05
        if accepted:
            final[final.index(old)] = new
        trace.append({"primary": old, "reserve": new, "region": region, "accepted": accepted})
    if len(final) != 128 or len(set(final)) != 128:
        raise RuntimeError("replacement must preserve unique K128")
    return torch.tensor(final, dtype=torch.long), trace
