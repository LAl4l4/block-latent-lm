import json
from pathlib import Path
import random
import torch
from tokenizers import Tokenizer
from latent_ar.model import LatentARModel, ModelConfig
from latent_ar.data import StoryDataset
from train import precision_context

checkpoint = torch.load('runs/overfit/last.pt', map_location='cpu', weights_only=True, mmap=True)
model = LatentARModel(ModelConfig(**checkpoint['model_config'])).cuda().eval()
model.load_state_dict(checkpoint['model'])
tokenizer = Tokenizer.from_file('runs/overfit/tokenizer.json')
text = 'Someone just find a ghost. And the ghost scare them!'
ids = tokenizer.encode(text, add_special_tokens=False).ids
prompt = torch.tensor([ids], device='cuda')
lengths = torch.tensor([len(ids)], device='cuda')
bos = torch.tensor([[model.cfg.bos_id]], device='cuda')
autocast = precision_context(torch.device('cuda'), 'bf16')
print(json.dumps({'saved_step':checkpoint['step'], 'prompt_tokens':len(ids), 'training':checkpoint['config']['training']}))
previous = None
with torch.inference_mode(), autocast():
    for steps in [1, 16, 256]:
        memory, positions, valid, _ = model.rollout(prompt, lengths, steps)
        memory_kvs = model.decoder.project_memory(memory, positions)
        logits, _ = model.decoder(bos, valid, steps, memory_kvs)
        scores = logits[:, -1].float()
        scores[:, model.cfg.pad_id] = -torch.inf
        scores[:, model.cfg.bos_id] = -torch.inf
        raw_prob = scores.softmax(-1)
        sampled_scores = scores / 0.8
        threshold = sampled_scores.topk(50, dim=-1).values[:, -1:]
        sampled_scores = sampled_scores.masked_fill(sampled_scores < threshold, -torch.inf)
        prob = sampled_scores.softmax(-1)
        chosen = torch.multinomial(prob, 1, generator=torch.Generator(device='cuda').manual_seed(42)).item()
        values, indices = raw_prob[0].topk(8)
        print(json.dumps({'rollout_steps':steps, 'raw_eos_probability':raw_prob[0,2].item(), 'sampling_eos_probability':prob[0,2].item(), 'seed42_first_id':chosen, 'seed42_first_token':tokenizer.id_to_token(chosen), 'top':[(tokenizer.id_to_token(i),p) for i,p in zip(indices.tolist(), values.tolist())]}))
    for seed in [42,43,44]:
        output = model.generate(prompt,lengths,max_new_tokens=64,generator=torch.Generator(device='cuda').manual_seed(seed))
        print(json.dumps({'seed':seed, 'ids':output[0].tolist(), 'text':tokenizer.decode(output[0].tolist())}))
    output = model.generate(prompt,lengths,max_new_tokens=64,temperature=0)
    print(json.dumps({'greedy':tokenizer.decode(output[0].tolist()), 'ids':output[0].tolist()}))

training = checkpoint['config']['training']
data = StoryDataset(Path('data/prepared'),'train',training['min_prompt_tokens'],training['max_prompt_tokens'],training['max_target_tokens'],0,1,limit=1000)
rng=random.Random(42)
first_eos=0
short_targets=0
for i in range(10000):
    p,y=data.example(rng.randrange(data.size),rng)
    first_eos += int(y[0]==2)
    short_targets += int(len(y)<64)
print(json.dumps({'sampled_training_windows':10000, 'first_target_eos':first_eos, 'target_shorter_than_64':short_targets}))
