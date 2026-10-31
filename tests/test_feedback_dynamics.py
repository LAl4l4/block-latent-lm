import torch
from latent_ar.calm_token import CalmTokenConfig, CalmTokenModel
from experiments.feedback_dynamics import prepare_feedback_history, next_block_logits, behavior_distillation_loss


def model():
    torch.manual_seed(7)
    return CalmTokenModel(CalmTokenConfig(vocab_size=32, chunk_size=4, encoder_dim=8, dim=16, layers=1, heads=2, ff_dim=32, decoder_dim=8, decoder_layers=1, decoder_heads=2, decoder_ff_dim=16, token_head_dim=8, noise_dim=4, energy_layers=1, energy_candidates=2, target_samples=2, energy_batch_blocks=4))


def test_variable_history_and_gradients():
    teacher = model()
    prompts = torch.tensor([[1, 4, 5, 6, 7, 8], [1, 9, 0, 0, 0, 0]])
    lengths = torch.tensor([6, 2])
    past = torch.randn(2, 2, 16)
    past_lengths = torch.tensor([1, 2])
    tokens = torch.tensor([[3, 4, 2, -100], [5, 2, -100, -100]])
    history = prepare_feedback_history(teacher, prompts, lengths, past, past_lengths)
    target = torch.randn(2, 16)
    joint = next_block_logits(teacher, history, target, tokens)
    for i in range(2):
        single = prepare_feedback_history(teacher, prompts[i:i+1, :lengths[i]], lengths[i:i+1], past[i:i+1], past_lengths[i:i+1])
        torch.testing.assert_close(joint[i], next_block_logits(teacher, single, target[i:i+1], tokens[i:i+1])[0], atol=1e-6, rtol=1e-5)
    loss, stats = behavior_distillation_loss(teacher, history, target, target, tokens)
    assert abs(float(stats['kl'])) < 1e-7 and float(loss) < 1e-7
    prediction = (target + torch.randn_like(target)).requires_grad_()
    loss, stats = behavior_distillation_loss(teacher, history, prediction, target, tokens, mse_weight=0)
    loss.backward()
    assert prediction.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert int(stats['valid_tokens']) == 5
    altered = tokens.clone(); altered[0, 3] = 7
    # The final ignored token cannot affect preceding predictions.
    torch.testing.assert_close(joint[0, :3], next_block_logits(teacher, history, target, altered)[0, :3])


def test_cuda_bf16_frozen_eval_gru_backward():
    import pytest
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA with bf16 is unavailable")
    teacher = model().cuda()
    prompts = torch.tensor([[1, 4, 5, 6]], device="cuda")
    lengths = torch.tensor([4], device="cuda")
    past = torch.empty(1, 0, 16, device="cuda")
    target = torch.randn(1, 16, device="cuda")
    tokens = torch.tensor([[3, 4, 2, -100]], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        history = prepare_feedback_history(teacher, prompts, lengths, past, torch.zeros_like(lengths))
        same, stats = behavior_distillation_loss(teacher, history, target, target, tokens)
        assert abs(float(stats["kl"])) < 1e-6
        prediction = (target + torch.randn_like(target)).requires_grad_()
        loss, _ = behavior_distillation_loss(teacher, history, prediction, target, tokens, mse_weight=0)
    loss.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
