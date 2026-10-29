"""阶段二：冻结合格的 codec，用 Energy loss 训练下一块 latent 分布。"""

from latent_ar.staged_training import CodecQualityError, run_stage


if __name__ == "__main__":
    try:
        run_stage("prior")
    except CodecQualityError as error:
        raise SystemExit(f"[重建未达标] {error}") from None
