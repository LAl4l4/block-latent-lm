"""冻结合格的 VAE，训练历史压缩器、主干及随机生成头。"""

from latent_ar.calm_training import run

if __name__ == "__main__":
    run("calm")
