"""阶段一：训练四 token 块的精确重建，运行后先检查 token / 整块准确率。"""

from latent_ar.staged_training import run_stage


if __name__ == "__main__":
    run_stage("codec")
