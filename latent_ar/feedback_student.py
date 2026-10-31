"""蒸馏文本反馈：从主干状态和教师噪声直接预测历史压缩向量。"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class StudentConfig:
    context_dim: int = 512
    noise_dim: int = 0
    hidden_dim: int = 512
    layers: int = 2
    feedback_dim: int = 512

    def __post_init__(self):
        if (
            min(self.context_dim, self.hidden_dim, self.layers, self.feedback_dim) < 1
            or self.noise_dim < 0
        ):
            raise ValueError("学生维度和层数必须为正；噪声维度可为零")


class FeedbackStudent(nn.Module):
    """预测 G(D(VAEmean(GRU(h,u))))；输入 h 是上下文，不是内容 latent。"""

    architecture = "feedback_student_v1"

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.context_norm = nn.LayerNorm(cfg.context_dim)
        self.context = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.noise = nn.Linear(cfg.noise_dim, cfg.hidden_dim) if cfg.noise_dim else None
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            for _ in range(cfg.layers)
        )
        self.output = nn.Linear(cfg.hidden_dim, cfg.feedback_dim)

    def forward(self, context, noise=None):
        if context.shape[-1] != self.cfg.context_dim:
            raise ValueError("上下文维度与学生配置不匹配")
        x = self.context(self.context_norm(context))
        if self.noise is None:
            if noise is not None:
                raise ValueError("确定性学生不接收噪声")
        else:
            if noise is None or noise.shape != (
                *context.shape[:-1],
                self.cfg.noise_dim,
            ):
                raise ValueError("需要与上下文批次匹配的教师均匀噪声")
            x = x + self.noise(noise - 0.5)
        for block in self.blocks:
            x = x + block(x)
        return self.output(x)

    def checkpoint(self):
        return {
            "architecture": self.architecture,
            "config": asdict(self.cfg),
            "model": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if checkpoint.get("architecture") != cls.architecture:
            raise ValueError("不是反馈学生检查点")
        instance = cls(StudentConfig(**checkpoint["config"]))
        instance.load_state_dict(checkpoint["model"], strict=True)
        return instance


@dataclass
class AttentionStudentConfig:
    context_dim: int = 512
    feedback_dim: int = 512
    hidden_dim: int = 256
    layers: int = 2
    heads: int = 4
    history_blocks: int = 2
    noise_dim: int = 0
    use_prompt_history: bool = False

    def __post_init__(self):
        if (
            min(
                self.context_dim,
                self.feedback_dim,
                self.hidden_dim,
                self.layers,
                self.heads,
            )
            < 1
            or self.history_blocks < 0
            or self.hidden_dim % self.heads
            or self.noise_dim != 0
        ):
            raise ValueError("无效的注意力学生配置")


class AttentionFeedbackStudent(nn.Module):
    """仅使用当前和过去的主干状态，窗口右端始终是当前状态。"""

    architecture = "feedback_student_attention_v1"

    def __init__(self, cfg: AttentionStudentConfig):
        super().__init__()
        self.cfg = cfg
        self.context_norm = nn.LayerNorm(cfg.context_dim)
        self.context = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.position = nn.Parameter(
            torch.zeros(cfg.history_blocks + 1, cfg.hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            cfg.hidden_dim,
            cfg.heads,
            cfg.hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, cfg.layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output = nn.Linear(cfg.hidden_dim, cfg.feedback_dim)

    def forward(self, context, noise=None, valid=None):
        if noise is not None:
            raise ValueError("确定性学生不接收噪声")
        if context.ndim != 3 or context.shape[-1] != self.cfg.context_dim:
            raise ValueError("注意力学生需要批次、窗口和上下文维度")
        if context.shape[1] != self.cfg.history_blocks + 1:
            raise ValueError("窗口长度与学生配置不匹配")
        if (
            valid is None
            or valid.shape != context.shape[:2]
            or not bool(valid[:, -1].all())
        ):
            raise ValueError("窗口掩码必须包含每个样本的当前状态")
        # 清除 padding 的值，避免无效位置的 NaN 或极值污染计算。
        context = context.masked_fill(~valid[..., None], 0)
        x = self.context(self.context_norm(context)) + self.position
        # 每次输入都截止当前块；只读取窗口最右端输出，因此不会访问未来状态。
        x = self.encoder(x, src_key_padding_mask=~valid)
        return self.output(self.output_norm(x[:, -1]))

    def checkpoint(self):
        return {
            "architecture": self.architecture,
            "config": asdict(self.cfg),
            "model": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if checkpoint.get("architecture") != cls.architecture:
            raise ValueError("不是注意力反馈学生检查点")
        instance = cls(AttentionStudentConfig(**checkpoint["config"]))
        instance.load_state_dict(checkpoint["model"], strict=True)
        return instance


def load_feedback_student(checkpoint):
    if checkpoint.get("architecture") == FeedbackStudent.architecture:
        return FeedbackStudent.from_checkpoint(checkpoint)
    if checkpoint.get("architecture") == AttentionFeedbackStudent.architecture:
        return AttentionFeedbackStudent.from_checkpoint(checkpoint)
    raise ValueError("未知反馈学生结构")


def build_history_windows(data, history_blocks):
    """按故事及块号重建窗口，轨迹开始前的位置作为左侧 padding。"""
    if history_blocks < 0:
        raise ValueError("历史块数不能为负")
    h, stories, blocks = data["h"], data["story_id"], data["block_index"]
    if len(h) != len(stories) or len(h) != len(blocks):
        raise ValueError("轨迹字段长度不一致")
    lookup = {}
    story_blocks = {}
    for i, (story, block) in enumerate(zip(stories.tolist(), blocks.tolist())):
        key = (story, block)
        if key in lookup or block < 0:
            raise ValueError("故事块编号重复或无效")
        lookup[key] = i
        story_blocks.setdefault(story, []).append(block)
    for positions in story_blocks.values():
        positions.sort()
        if positions != list(range(len(positions))):
            raise ValueError("故事块编号必须从零连续")
    indices = torch.zeros((len(h), history_blocks + 1), dtype=torch.long)
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for i, (story, block) in enumerate(zip(stories.tolist(), blocks.tolist())):
        for offset in range(history_blocks + 1):
            index = lookup.get((story, block - history_blocks + offset))
            if index is not None:
                indices[i, offset], valid[i, offset] = index, True
    return indices, valid


def prompt_history(hidden, n_history, history_blocks):
    """取得每行当前预填充状态之前的历史，左侧 padding。"""
    offsets = torch.arange(history_blocks, 0, -1, device=hidden.device)
    positions = n_history[:, None] - 1 - offsets[None, :]
    valid = positions >= 0
    rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    values = hidden[rows, positions.clamp_min(0)]
    return values.masked_fill(~valid[..., None], 0), valid


def materialize_history_windows(data, history_blocks, use_prompt_history=False):
    """重建训练窗口；负块号来自对应故事的提示词预填充。"""
    indices, valid = build_history_windows(data, history_blocks)
    window = data["h"][indices].clone()
    window.masked_fill_(~valid[..., None], 0)
    if not use_prompt_history:
        return window, valid
    if not all(key in data for key in ("prefix_h", "prefix_valid", "prompt_story_ids")):
        raise ValueError("提示词历史数据缺失")
    prefix, prefix_valid = data["prefix_h"], data["prefix_valid"]
    ids = data["prompt_story_ids"].tolist()
    if (
        prefix.ndim != 3
        or prefix.shape[0] != len(ids)
        or prefix.shape[2] != data["h"].shape[1]
        or prefix_valid.shape != prefix.shape[:2]
        or prefix_valid.dtype != torch.bool
        or len(set(ids)) != len(ids)
        or prefix.shape[1] < history_blocks
    ):
        raise ValueError("提示词历史数据格式或窗口长度错误")
    story_rows = {story: i for i, story in enumerate(ids)}
    for i, (story, block) in enumerate(
        zip(data["story_id"].tolist(), data["block_index"].tolist())
    ):
        if story not in story_rows:
            raise ValueError("轨迹故事没有对应提示词历史")
        for offset in range(history_blocks + 1):
            block_number = block - history_blocks + offset
            if block_number < 0:
                prefix_index = prefix.shape[1] + block_number
                if prefix_index >= 0 and prefix_valid[story_rows[story], prefix_index]:
                    window[i, offset] = prefix[story_rows[story], prefix_index]
                    valid[i, offset] = True
    return window, valid


def feedback_statistics(prediction, target, active=None):
    """MSE 用于优化，余弦相似度只用于诊断。"""
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("反馈预测与目标形状必须相同")
    if active is not None:
        if active.shape != prediction.shape[:-1] or not bool(active.any()):
            raise ValueError("active 形状错误或没有有效样本")
        prediction, target = prediction[active], target[active]
    error = F.mse_loss(prediction.float(), target.float())
    cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1).mean()
    return {"loss": error, "mse": error.detach(), "cosine": cosine.detach()}


@torch.inference_mode()
def generate_student_feedback(
    teacher,
    student,
    prompt,
    lengths,
    max_new_tokens=64,
    generator=None,
    return_trace=False,
    min_new_tokens=0,
):
    """学生反馈推进主干；同时记录该同一学生历史下真实教师反馈。"""
    if max_new_tokens < 1 or not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("Invalid generation lengths")
    if student.cfg.noise_dim == 0 and teacher.cfg.token_temperature != 0:
        raise ValueError("确定性学生要求教师 token_temperature=0")
    if student.cfg.noise_dim not in (0, teacher.cfg.chunk_size):
        raise ValueError("随机学生噪声维度必须等于教师每块 token 数")
    if (
        student.cfg.context_dim != teacher.cfg.dim
        or student.cfg.feedback_dim != teacher.cfg.dim
    ):
        raise ValueError("学生输入和反馈维度必须匹配教师主干")
    teacher_was, student_was = teacher.training, student.training
    teacher.eval()
    student.eval()
    try:
        latent_rng, _ = teacher.generation_generators(generator, prompt.device)
        encoded, valid = teacher.encode_history(prompt, lengths)
        batch, width, _ = encoded.shape
        n_history = valid.sum(-1)
        positions = torch.arange(width, device=prompt.device).expand(batch, -1)
        hidden, cache = teacher.backbone(encoded, positions, valid)
        context = hidden[torch.arange(batch, device=prompt.device), n_history - 1]
        finished = torch.zeros(batch, dtype=torch.bool, device=prompt.device)
        parts = []
        records = [] if return_trace else None
        history_window = None
        history_valid = None
        if isinstance(student, AttentionFeedbackStudent):
            history_blocks = student.cfg.history_blocks
            if student.cfg.use_prompt_history:
                history_window, history_valid = prompt_history(
                    hidden, n_history, history_blocks
                )
            else:
                history_window = context.new_zeros(
                    (batch, history_blocks, context.shape[-1])
                )
                history_valid = torch.zeros(
                    (batch, history_blocks), dtype=torch.bool, device=context.device
                )
        for start in range(0, max_new_tokens, teacher.cfg.chunk_size):
            active = ~finished
            noise = teacher.draw_noise((batch,), prompt.device, latent_rng)
            z = teacher.sample_latent(context, noise=noise)
            width = min(teacher.cfg.chunk_size, max_new_tokens - start)
            logits = teacher.codec.decode(z[:, None])[:, :width].float().clone()
            logits[..., teacher.cfg.pad_id] = -torch.inf
            logits[..., teacher.cfg.bos_id] = -torch.inf
            index = torch.arange(start, start + width, device=prompt.device)
            logits[:, index < min_new_tokens, teacher.cfg.eos_id] = -torch.inf
            chosen = logits.argmax(-1)
            actual_lengths = torch.zeros(batch, dtype=torch.long, device=prompt.device)
            emitted = []
            for slot in range(width):
                alive = ~finished
                word = torch.where(alive, chosen[:, slot], teacher.cfg.eos_id)
                emitted.append(word)
                actual_lengths += alive.long()
                finished |= alive & word.eq(teacher.cfg.eos_id)
                if bool(finished.all()):
                    break
            words = torch.stack(emitted, dim=1)
            parts.append(words)
            # 真实反馈严格按 decoder 实际输出及 EOS 结束位置编码。
            rows = active.nonzero(as_tuple=True)[0]
            actual = words[rows].clone()
            mask = (
                torch.arange(actual.shape[1], device=actual.device)[None]
                >= actual_lengths[rows, None]
            )
            actual.masked_fill_(mask, teacher.cfg.pad_id)
            true_encoded, _ = teacher.encode_history(actual, actual_lengths[rows])
            true_feedback = true_encoded.new_zeros((batch, student.cfg.feedback_dim))
            true_feedback[rows] = true_encoded[:, 0]
            if isinstance(student, AttentionFeedbackStudent):
                window = torch.cat((history_window, context[:, None]), dim=1)
                window_valid = torch.cat(
                    (
                        history_valid,
                        torch.ones((batch, 1), dtype=torch.bool, device=context.device),
                    ),
                    dim=1,
                )
                history_window = window[:, 1:]
                history_valid = window_valid[:, 1:]
                predicted = student(window, valid=window_valid)
            else:
                predicted = student(context, noise if student.cfg.noise_dim else None)
            if return_trace:
                records.append(
                    {
                        "context": context.detach().clone(),
                        "noise": noise.detach().clone(),
                        "z": z.detach().clone(),
                        "true_feedback": true_feedback.detach().clone(),
                        "predicted_feedback": predicted.detach().clone(),
                        "token_ids": words.detach().clone(),
                        "lengths": actual_lengths.detach().clone(),
                        "active": active.detach().clone(),
                    }
                )
            if bool(finished.all()) or start + width >= max_new_tokens:
                break
            valid = torch.cat((valid, (~finished)[:, None]), dim=1)
            hidden, cache = teacher.backbone(
                predicted[:, None],
                (n_history + start // teacher.cfg.chunk_size)[:, None],
                valid,
                cache,
            )
            context = hidden[:, 0]
        output = torch.cat(parts, 1)
        return {"output_ids": output, "records": records} if return_trace else output
    finally:
        teacher.train(teacher_was)
        student.train(student_was)
