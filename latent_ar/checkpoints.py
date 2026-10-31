"""区分新旧 decoder 的权重格式，让历史实验仍可生成和比较。"""

BLOCK_ARCHITECTURE = "latent_block_v2"
LEGACY_ARCHITECTURE = "latent_ar_v1"
CODEC_ARCHITECTURE = "block_codec_v1"
PRIOR_ARCHITECTURE = "latent_codec_ar_v3"
FLOW_ARCHITECTURE = "latent_codec_flow_v3"
ENERGY_ARCHITECTURE = "latent_codec_energy_v4"
MIXTURE_ARCHITECTURE = "latent_codec_mixture_v5"
TOKEN_HEAD_ARCHITECTURE = "latent_codec_token_head_v6"
ENERGY_TEXT_ARCHITECTURE = "latent_codec_energy_text_v5"
DENOISING_ARCHITECTURE = "latent_codec_denoising_v7"
VAE_ARCHITECTURE = "block_vae_v1"
CALM_ARCHITECTURE = "latent_calm_v1"
CONDITIONAL_FLOW_ARCHITECTURE = "latent_conditional_flow_v1"
CALM_TOKEN_ARCHITECTURE = "latent_calm_token_v1"
TRAINABLE_PRIOR_ARCHITECTURES = {
    PRIOR_ARCHITECTURE,
    ENERGY_ARCHITECTURE,
    MIXTURE_ARCHITECTURE,
    DENOISING_ARCHITECTURE,
    TOKEN_HEAD_ARCHITECTURE,
}


def checkpoint_architecture(checkpoint):
    """旧检查点没有架构标记，其 decoder 是带 cross-attention 的 AR 模型。"""
    architecture = checkpoint.get("architecture")
    if architecture is None:
        decoder_type = checkpoint["model_config"].get("decoder_type")
        return (
            BLOCK_ARCHITECTURE
            if decoder_type == "block_parallel"
            else LEGACY_ARCHITECTURE
        )
    if architecture not in {
        BLOCK_ARCHITECTURE,
        LEGACY_ARCHITECTURE,
        CODEC_ARCHITECTURE,
        PRIOR_ARCHITECTURE,
        FLOW_ARCHITECTURE,
        ENERGY_ARCHITECTURE,
        MIXTURE_ARCHITECTURE,
        TOKEN_HEAD_ARCHITECTURE,
        ENERGY_TEXT_ARCHITECTURE,
        DENOISING_ARCHITECTURE,
        VAE_ARCHITECTURE,
        CALM_ARCHITECTURE,
        CONDITIONAL_FLOW_ARCHITECTURE,
        CALM_TOKEN_ARCHITECTURE,
    }:
        raise ValueError(f"Unknown checkpoint architecture: {architecture}")
    return architecture


def load_model(checkpoint, device):
    """根据检查点建立对应模型，严格加载全部权重，不做静默迁移。"""
    architecture = checkpoint_architecture(checkpoint)
    if architecture == BLOCK_ARCHITECTURE:
        from latent_ar.model import LatentARModel, ModelConfig
    elif architecture == CODEC_ARCHITECTURE:
        from latent_ar.codec import BlockCodec as LatentARModel
        from latent_ar.model import ModelConfig
    elif architecture == PRIOR_ARCHITECTURE:
        from latent_ar.staged import LatentPriorModel as LatentARModel
        from latent_ar.staged import StagedModelConfig as ModelConfig
    elif architecture == ENERGY_ARCHITECTURE:
        from latent_ar.energy import LatentEnergyModel as LatentARModel
        from latent_ar.energy import EnergyModelConfig as ModelConfig
    elif architecture == FLOW_ARCHITECTURE:
        from latent_ar.legacy_flow_prior import LatentPriorModel as LatentARModel
        from latent_ar.legacy_flow_prior import StagedModelConfig as ModelConfig
    elif architecture == MIXTURE_ARCHITECTURE:
        from latent_ar.mixture import LatentMixtureModel as LatentARModel
        from latent_ar.mixture import MixtureModelConfig as ModelConfig
    elif architecture == TOKEN_HEAD_ARCHITECTURE:
        from latent_ar.token_head import LatentTokenHeadModel as LatentARModel
        from latent_ar.token_head import TokenHeadConfig as ModelConfig
    elif architecture == ENERGY_TEXT_ARCHITECTURE:
        from latent_ar.energy_text import LatentEnergyTextModel as LatentARModel
        from latent_ar.energy_text import EnergyTextConfig as ModelConfig
    elif architecture == DENOISING_ARCHITECTURE:
        from latent_ar.denoising import LatentDenoisingModel as LatentARModel
        from latent_ar.denoising import DenoisingConfig as ModelConfig
    elif architecture == VAE_ARCHITECTURE:
        from latent_ar.vae import BlockVAE as LatentARModel
        from latent_ar.vae import VAEConfig as ModelConfig
    elif architecture == CALM_ARCHITECTURE:
        from latent_ar.calm import CalmModel as LatentARModel
        from latent_ar.calm import CalmConfig as ModelConfig
    elif architecture == CONDITIONAL_FLOW_ARCHITECTURE:
        from latent_ar.conditional_flow import ConditionalFlowModel as LatentARModel
        from latent_ar.conditional_flow import FlowConfig as ModelConfig
    elif architecture == CALM_TOKEN_ARCHITECTURE:
        from latent_ar.calm_token import CalmTokenModel as LatentARModel
        from latent_ar.calm_token import CalmTokenConfig as ModelConfig
    else:
        from latent_ar.legacy_model import LatentARModel, ModelConfig

    model = LatentARModel(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model
