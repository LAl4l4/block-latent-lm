"""加载检查点与配套词表，执行第一阶段的同步文本续写。"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from latent_ar.checkpoints import (
    BLOCK_ARCHITECTURE,
    CODEC_ARCHITECTURE,
    VAE_ARCHITECTURE,
    CALM_ARCHITECTURE,
    CONDITIONAL_FLOW_ARCHITECTURE,
    CALM_TOKEN_ARCHITECTURE,
    FLOW_ARCHITECTURE,
    ENERGY_ARCHITECTURE,
    MIXTURE_ARCHITECTURE,
    ENERGY_TEXT_ARCHITECTURE,
    DENOISING_ARCHITECTURE,
    TOKEN_HEAD_ARCHITECTURE,
    PRIOR_ARCHITECTURE,
    checkpoint_architecture,
    load_model,
)
from train import precision_context, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=0,
        help="达到该长度前屏蔽 EOS；默认 0，允许模型自行决定停止",
    )
    parser.add_argument(
        "--show-info", action="store_true", help="显示生成长度和停止原因"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="0 为贪心选词；Energy latent 仍由 seed 随机采样",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--flow-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--latent-temperature", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feedback-mode", choices=["text", "raw"],
        help="有限混合头的逐块反馈对照：text 回灌实际文本编码，raw 回灌预测 latent；均要求 temperature=0",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 0 <= args.min_new_tokens <= args.max_new_tokens:
        parser.error("需要满足 0 <= min-new-tokens <= max-new-tokens")
    # 配置、模型权重与 tokenizer 必须配套，不能换成另一个实验的词表。
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    device = select_device(args.device)
    architecture = checkpoint_architecture(checkpoint)
    if architecture in {CODEC_ARCHITECTURE, VAE_ARCHITECTURE}:
        if architecture == VAE_ARCHITECTURE:
            parser.error("这是 VAE 重建检查点。请先用 train_calm.py 训练主干，再生成续写。")
        parser.error(
            "这是 codec 重建检查点。请先用 train_prior.py 训练主干，再生成续写。"
        )
    if args.temperature is None:
        args.temperature = (
            0.0
            if architecture
            in {
                BLOCK_ARCHITECTURE,
                PRIOR_ARCHITECTURE,
                FLOW_ARCHITECTURE,
                ENERGY_ARCHITECTURE,
                DENOISING_ARCHITECTURE,
                TOKEN_HEAD_ARCHITECTURE,
                CALM_ARCHITECTURE,
                CONDITIONAL_FLOW_ARCHITECTURE,
                CALM_TOKEN_ARCHITECTURE,
            }
            else (
                1.0
                if architecture in {MIXTURE_ARCHITECTURE, ENERGY_TEXT_ARCHITECTURE}
                else 0.8
            )
        )
        if architecture == MIXTURE_ARCHITECTURE and (
            checkpoint["model_config"].get("feedback_mode") == "text"
            or args.feedback_mode is not None
        ):
            args.temperature = 0.0
    prior_options = {}
    if args.feedback_mode is not None:
        if architecture != MIXTURE_ARCHITECTURE:
            parser.error("--feedback-mode 当前仅支持有限混合头检查点")
        prior_options["feedback_mode"] = args.feedback_mode
    if architecture == MIXTURE_ARCHITECTURE and (
        checkpoint["model_config"].get("feedback_mode") == "text"
        or args.feedback_mode is not None
    ) and args.temperature != 0:
        parser.error("真实反馈实验固定 decoder 为 argmax，请使用 --temperature 0")
    if architecture == FLOW_ARCHITECTURE:
        if args.flow_steps is not None:
            prior_options["flow_steps"] = args.flow_steps
        if args.latent_temperature is not None:
            prior_options["latent_temperature"] = args.latent_temperature
    elif args.flow_steps is not None or args.latent_temperature is not None:
        parser.error("该检查点使用直接 latent 预测，不接受 flow 采样参数")
    tokenizer = Tokenizer.from_file(str(args.checkpoint.parent / "tokenizer.json"))
    prompt = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not prompt:
        parser.error("Prompt must contain text")
    trained_length = checkpoint["config"]["training"]["max_prompt_tokens"]
    # 当前原型没有验证超出训练长度的 prompt，默认保留最近的一段上下文。
    if len(prompt) > trained_length:
        print(
            f"Warning: keeping the last {trained_length} prompt tokens (training context limit)."
        )
        prompt = prompt[-trained_length:]
    model = load_model(checkpoint, device)
    autocast = precision_context(device, checkpoint["config"]["training"]["precision"])
    # 单独建立采样随机数，固定 seed 可以复查同一设置下的输出。
    with autocast():
        output = model.generate(
            torch.tensor([prompt], device=device),
            torch.tensor([len(prompt)], device=device),
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            use_latents=checkpoint["config"].get("ablation", "full") == "full",
            generator=torch.Generator(device=device).manual_seed(args.seed),
            min_new_tokens=args.min_new_tokens,
            **prior_options,
        )
    # EOS 是特殊 token，decode 默认隐藏它；单独报告可避免误以为程序没运行。
    generated_ids = output[0].tolist()
    stopped_by_eos = model.cfg.eos_id in generated_ids
    if stopped_by_eos:
        generated_ids = generated_ids[: generated_ids.index(model.cfg.eos_id)]
    continuation = tokenizer.decode(generated_ids)
    print(tokenizer.decode(prompt) + continuation)

    if not generated_ids:
        print(
            "[停止] 模型在第一个位置选择了 EOS，未生成续写。"
            "可用 --min-new-tokens 16 显式限制最短长度；"
            "该限制不代表模型质量改善。"
        )
    if args.show_info:
        reason = "EOS" if stopped_by_eos else "达到最大长度"
        print(f"[生成] {len(generated_ids)} 个新 token | 停止原因：{reason}")


if __name__ == "__main__":
    main()
