import torch

from experiments.diagnose_token_context import block_inputs, parameter_change, score_blocks, shuffled_contexts
from latent_ar.calm_token import CalmTokenConfig, CalmTokenModel


def test_shuffle_only_uses_other_valid_rows_at_same_block_position():
    contexts = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    shuffled, eligible = shuffled_contexts(contexts, valid, torch.Generator().manual_seed(10))
    assert eligible.tolist() == [[True, True, False, False], [True, True, False, False], [True, False, False, False]]
    for row, col in eligible.nonzero().tolist():
        assert not torch.equal(shuffled[row, col], contexts[row, col])
        assert any(torch.equal(shuffled[row, col], contexts[other, col])
                   for other in range(3) if other != row and valid[other, col])


def test_slot_counts_and_parameter_comparison():
    cfg = CalmTokenConfig(vocab_size=24, chunk_size=2, encoder_dim=8, dim=8,
                          layers=1, heads=2, ff_dim=16, decoder_dim=8,
                          decoder_layers=1, decoder_heads=2, decoder_ff_dim=16,
                          token_head_dim=8)
    model = CalmTokenModel(cfg)
    batch = {"prompt": torch.tensor([[1, 4], [1, 6]]), "prompt_lengths": torch.tensor([2, 2]),
             "labels": torch.tensor([[7, 8, 9], [10, 11, -100]])}
    contexts, blocks, previous, valid = block_inputs(model, batch)
    shuffled, eligible = shuffled_contexts(contexts, valid, torch.Generator().manual_seed(9))
    sums, counts = score_blocks(model, shuffled, blocks, previous, eligible)
    assert counts.tolist() == [2, 2]
    (sums.sum() / counts.sum()).backward()
    assert model.history_embedding.weight.grad is not None
    assert all(p.grad is None for p in model.codec.parameters())
    source = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        model.history_embedding.weight.add_(1)
    changes = parameter_change(model, source)
    assert abs(changes["history_embedding"]["delta_rms"] - 1) < 1e-6
    assert changes["backbone"]["delta_rms"] == 0
