from datetime import timedelta

import torch
import torch.nn as nn

from solaris.model.decoder import Perceiver3DDecoder
from solaris.model.encoder import Perceiver3DEncoder
from solaris.model.lora import LoRAMode
from solaris.model.swin3d import Swin3DTransformerBackbone

__all__ = ["Solaris_F107", "SolarisHighRes", "SolarisSmall", "SolarisTiny"]


class Solaris_F107(nn.Module):
    def __init__(
        self,
        freeze_backbone,
        output_dim,
        patch_size: int = 8,
        embed_dim: int = 256,
        encoder_depths: tuple[int, ...] = (2, 6, 2),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (2, 6, 2),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        window_size: tuple[int, int, int] = (2, 6, 12),
        max_history_size: int = 2,
        latent_levels: int = 4,
        out_levels: int = 8,
        enc_depth: int = 1,
        dec_depth: int = 1,
        num_heads: int = 16,
        drop_rate: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-5,
        use_lora: bool = False,
        lora_mode: LoRAMode = "single",
        lora_steps: int = 40,
        hidden_layer_dims=[512,512,512],
        dropout=0.1,
        mask_ratio=0.0,
    ):
        super().__init__()
        self.out_levels = out_levels

        self.encoder = Perceiver3DEncoder(
            patch_size=patch_size,
            embed_dim=embed_dim,
            max_history_size=max_history_size,
            latent_levels=latent_levels,
            depth=enc_depth,
            head_dim=embed_dim // num_heads,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            perceiver_ln_eps=perceiver_ln_eps,
        )

        self.backbone = Swin3DTransformerBackbone(
            embed_dim=embed_dim,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path,
            use_lora=use_lora,
            lora_mode=lora_mode,
            lora_steps=lora_steps,
        )

        if freeze_backbone:
            self.encoder.requires_grad_(False)
            self.backbone.requires_grad_(False)
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.required_grad = False
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.required_grad = False



        self.norm = nn.LayerNorm(embed_dim * 4)

        self.flatten = nn.Flatten(1, -1)

        # Define the dimensions of the MLP layers
        dims = [embed_dim * 4] + hidden_layer_dims
        #dims = [embed_dim/patch_size * 2] + hidden_layer_dims

        # Define the dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Define the fully connected layers
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        # Define the activation function
        #self.acts = nn.ModuleList([nn.LeakyReLU(0.01) for _ in range(len(dims) - 1)])
        self.acts = nn.ModuleList([nn.GELU() for _ in range(len(dims) - 1)])

        # Define the output layer
        self.fc_out = nn.Linear(dims[-1], output_dim)

        # Define the loss function
        self.criterion = nn.MSELoss()

        # Initialize a dictionary to store test predictions
        self.test_preds = {}

        self.register_buffer("means", torch.zeros(out_levels))
        self.register_buffer("stds", torch.ones(out_levels))

    def set_normalisation(self, means, stds):
        assert means.shape == (self.out_levels,) and stds.shape == (self.out_levels,)
        self.means.copy_(means)
        self.stds.copy_(stds)

    def normalise(self, x):
        if self.means.numel() != x.shape[2]:
            return x
        means = self.means.view(1, 1, -1, 1, 1)
        stds = self.stds.view(1, 1, -1, 1, 1)
        return (x - means) / stds



    def forward(self, x, metadata, lead_time, rollout_step):
        lead_time = lead_time if isinstance(lead_time, timedelta) else timedelta(hours=lead_time)
        x = self.normalise(x)

        H, W = x.size(-2), x.size(-1)
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        x = self.encoder(x, metadata, lead_time)

        x = self.backbone(x, lead_time, rollout_step=rollout_step, patch_res=patch_res)


        x_avg = x.mean(dim=1)
        x_max = x.max(dim=1).values
        x = torch.cat([x_avg, x_max], dim=-1)
        #x = x.flatten()

        x = self.norm(x)
        
        for fc, act in zip(self.fcs, self.acts):
            x = self.dropout(x)
            x = fc(x)
            x = act(x)

        logits = self.fc_out(x)

        return logits




