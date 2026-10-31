"""先训练 16-token VAE：重建、后验采样、KL 和训练期 dropout。"""

from latent_ar.calm_training import run

if __name__ == "__main__":
    run("vae")
