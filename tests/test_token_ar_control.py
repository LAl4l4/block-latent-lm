import random

import torch

from latent_ar.model import ModelConfig
from experiments.token_ar_baseline import TokenAR, packed_batch


def test_token_causality_and_cached_generation_agree():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    model = TokenAR(ModelConfig(vocab_size=32, dim=32, layers=2, heads=4, ff_dim=64))
    ids = torch.tensor([[4, 5, 6, 7]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    full, _ = model(ids, valid)
    changed = ids.clone()
    changed[:, 3] = 9
    other, _ = model(changed, valid)
    torch.testing.assert_close(full[:, :3], other[:, :3])
    _, cache = model(ids[:, :3], valid[:, :3])
    cached, _ = model(ids[:, 3:], valid, torch.tensor([[3]]), cache)
    torch.testing.assert_close(full[:, 3:], cached, atol=1e-6, rtol=1e-5)


def test_packing_supervises_first_target_and_never_prompt_or_padding():
    class Dataset:
        pad_id = 0

        def batch(self, rng, count):
            return {
                "prompt": torch.tensor([[4, 5, 6], [7, 8, 0]]),
                "prompt_lengths": torch.tensor([3, 2]),
                "labels": torch.tensor([[9, 10], [11, -100]]),
            }

    ids, labels, valid = packed_batch(
        Dataset(), random.Random(1), 2, torch.device("cpu")
    )
    assert ids.tolist() == [[4, 5, 6, 9], [7, 8, 0, 0]]
    assert labels.tolist() == [[-100, -100, 9, 10], [-100, 11, -100, -100]]
    assert valid.tolist() == [[True, True, True, True], [True, True, False, False]]
