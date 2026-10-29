"""只读诊断正在训练的 V2 检查点，不更新模型或训练文件。"""
import json
from pathlib import Path
import random
import math
import torch
import torch.nn.functional as F
import numpy as np
from tokenizers import Tokenizer
from latent_ar.checkpoints import load_model
from latent_ar.data import StoryDataset
from train import precision_context

torch.set_num_threads(2)
path=Path('runs/train-block-v2/last.pt')
# 不用 mmap，以免 Windows 上长期持有 last.pt 映射干扰训练进程替换文件。
checkpoint=torch.load(path,map_location='cpu',weights_only=True)
model=load_model(checkpoint,torch.device('cuda')).eval()
training=checkpoint['config']['training']
step=checkpoint['step']
tokenizer=Tokenizer.from_file(str(path.parent/'tokenizer.json'))
kwargs=dict(directory=Path('data/prepared'),min_prompt=training['min_prompt_tokens'],max_prompt=training['max_prompt_tokens'],max_target=64,pad_id=model.cfg.pad_id)
train=StoryDataset(split='train',limit=training.get('train_limit'),**kwargs)
dev=StoryDataset(split='dev',**kwargs)
rng=random.Random(20261029)
counts=np.ones(model.cfg.vocab_size,dtype=np.float64)
for _ in range(4096):
    _,y=train.example(rng.randrange(train.size),rng)
    counts += np.bincount(y,minlength=model.cfg.vocab_size)
log_unigram=torch.tensor(np.log(counts/counts.sum()),device='cuda',dtype=torch.float32)
variants=['original','temporal_mean','wrong_prompt','zero']
summary={v:{'loss':0.,'tokens':0,'correct':0,'groups':{str(g):[0.,0] for g in range(4)}} for v in variants}
unigram_loss=0.
cosine_temporal=[]
cosine_cross_prompt=[]
first=[]
autocast=precision_context(torch.device('cuda'),'bf16')
with torch.inference_mode():
    for batch in dev.evaluation_batches(8,16,training['seed']+10000):
        batch={k:v.cuda() for k,v in batch.items()}
        labels=batch['labels']
        valid=labels!=-100
        length=labels.shape[1]
        with autocast():
            z=model.rollout(batch['prompt'],batch['prompt_lengths'],math.ceil(length/4))
        z_float=z.float()
        cosine_temporal.append(float(F.cosine_similarity(z_float[:,1:],z_float[:,:-1],dim=-1).mean()))
        cosine_cross_prompt.append(float(F.cosine_similarity(z_float,z_float.roll(1,0),dim=-1).mean()))
        modified={'original':z,'temporal_mean':z.mean(1,keepdim=True).expand_as(z),'wrong_prompt':z.roll(1,0),'zero':torch.zeros_like(z)}
        for variant,latent in modified.items():
            with autocast():
                logits=model.decode(latent,length)
            losses=F.cross_entropy(logits.float().flatten(0,1),labels.flatten(),ignore_index=-100,reduction='none').reshape_as(labels)
            current=summary[variant]
            current['loss'] += float(losses.sum())
            current['tokens'] += int(valid.sum())
            current['correct'] += int(((logits.argmax(-1)==labels)&valid).sum())
            positions=torch.arange(length,device='cuda')
            for group,(begin,end) in enumerate([(0,4),(4,16),(16,32),(32,64)]):
                mask=valid & ((positions>=begin)&(positions<end))[None,:]
                current['groups'][str(group)][0]+=float(losses[mask].sum())
                current['groups'][str(group)][1]+=int(mask.sum())
            if not first and variant=='original':
                scores=logits[0,0].float()
                values,ids=scores.softmax(-1).topk(8)
                first=[(tokenizer.id_to_token(i),p) for i,p in zip(ids.tolist(),values.tolist())]
        unigram_loss += float((-log_unigram[labels.masked_fill(~valid,0)])[valid].sum())

metrics={}
for name,values in summary.items():
    ce=values['loss']/values['tokens']
    metrics[name]={'ce':ce,'ppl':math.exp(ce),'top1_accuracy':values['correct']/values['tokens'],'tokens':values['tokens'],'position_group_ce':{k:v[0]/v[1] for k,v in values['groups'].items()}}
metrics['unigram']={'ce':unigram_loss/summary['original']['tokens']}
# 一个小批次确认实际梯度存在，不做参数更新。
model.train()
batch=next(dev.evaluation_batches(2,1,training['seed']+10000))
batch={k:v.cuda() for k,v in batch.items()}
with autocast():
    logits=model(batch['prompt'],batch['prompt_lengths'],batch['labels'].shape[1])
loss=F.cross_entropy(logits.float().flatten(0,1),batch['labels'].flatten(),ignore_index=-100)
loss.backward()
gradients={}
for name,module in [('encoder',model.encoder),('backbone',model.backbone),('latent_head',model.latent_head),('decoder',model.decoder)]:
    squared=sum(float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None)
    gradients[name]=math.sqrt(squared)
report={'checkpoint_step':step,'dev_stories':128,'metrics':metrics,'adjacent_latent_cosine':sum(cosine_temporal)/len(cosine_temporal),'different_prompt_latent_cosine':sum(cosine_cross_prompt)/len(cosine_cross_prompt),'first_position_top_tokens':first,'gradient_norms_single_dev_batch':gradients,'notes':['Same fixed dev windows and targets for all variants.','Wrong-prompt means rotating generated latent sequences across an 8-story batch.','Unigram is Laplace-smoothed and fitted on 4096 training windows.','Perturbations are out of distribution; loss changes diagnose reliance, not causal quality gain.']}
output=Path(f'runs/block-v2-diagnostics-step{step}.json')
output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
print('Report:',output,flush=True)
