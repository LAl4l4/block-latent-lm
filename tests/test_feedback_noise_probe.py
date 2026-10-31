import torch
from latent_ar.calm_token import CalmTokenConfig, CalmTokenModel
from experiments.feedback_noise_probe import draw_feedback


def test_same_noise_reproduces_words_and_feedback():
    torch.manual_seed(12)
    model = CalmTokenModel(CalmTokenConfig(vocab_size=32, chunk_size=4, encoder_dim=8, dim=16, layers=1, heads=2, ff_dim=32, decoder_dim=8, decoder_layers=1, decoder_heads=2, decoder_ff_dim=16, token_head_dim=8, noise_dim=4, energy_layers=1, energy_candidates=2, target_samples=2, energy_batch_blocks=4, token_temperature=0.8)).eval()
    h = torch.randn(3, 16)
    u = torch.rand(3, 4)
    a = draw_feedback(model, h, u)
    b = draw_feedback(model, h, u)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
    words, lengths, feedback, continuing = a
    assert not words.eq(model.cfg.bos_id).any()
    assert torch.equal(continuing, ~words.eq(model.cfg.eos_id).any(-1))
    for row in range(len(h)):
        assert words[row, lengths[row]:].eq(model.cfg.pad_id).all()
        expected, _ = model.encode_history(words[row:row+1], lengths[row:row+1])
        torch.testing.assert_close(feedback[row], expected[0, 0])
