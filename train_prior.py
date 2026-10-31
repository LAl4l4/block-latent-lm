"""阶段二：冻结合格的 codec，按配置训练下一块的条件分布。"""

from latent_ar.staged_training import CodecQualityError, run_stage


if __name__ == "__main__":
    try:
        run_stage("prior")
    except CodecQualityError as error:
        raise SystemExit(f"[重建未达标] {error}") from None
