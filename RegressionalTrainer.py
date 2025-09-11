from torch import nn
from typing import Optional
from encoder import get_image_size_from_processor
import torch

class VisionTextRegressor(nn.Module):
    def __init__(self, encoder, num_study_types, study_embed_dim=128,
             encoder_dim=None, processor=None, image_size=None,
             pool="pooler", freeze=True):
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.image_size = image_size

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # Infer encoder_dim if not given
        if encoder_dim is None:
            img_size = self.image_size or get_image_size_from_processor(self.processor, fallback=224)
            device = next(self.encoder.parameters()).device
            dummy = torch.randn(1, 3, img_size, img_size, device=device)
            with torch.no_grad():
                out = self.encoder(pixel_values=dummy) if "forward" in dir(self.encoder) else self.encoder(dummy)

            if hasattr(out, "pooler_output"):
                img = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                img = out.last_hidden_state[:, 0]
            elif hasattr(out, "image_embeds"):
                img = out.image_embeds
            else:
                raise ValueError("Unknown encoder output format")

            encoder_dim = img.shape[1]

        self.encoder_dim = encoder_dim

        # Study embeddings
        self.study_embedding = nn.Embedding(num_study_types, study_embed_dim)
        self.film_gamma = nn.Linear(study_embed_dim, encoder_dim)
        self.film_beta  = nn.Linear(study_embed_dim, encoder_dim)

        # Projections
        self.img_proj   = nn.Sequential(
            nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, 512), nn.ReLU()
        )
        self.study_proj = nn.Sequential(
            nn.LayerNorm(study_embed_dim), nn.Linear(study_embed_dim, 128), nn.ReLU()
        )

        # Regressor
        self.regressor  = nn.Sequential(
            nn.Linear(512+128, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, pixel_values, study_type_ids, labels: Optional[torch.FloatTensor] = None):
        with torch.no_grad():
            out = self.encoder(pixel_values=pixel_values)
            if isinstance(out, dict):
                img = out["pooler_output"] if "pooler_output" in out else out["last_hidden_state"][:, 0]
            else:
                img = out

        study = self.study_embedding(study_type_ids)
        # FiLM conditioning
        gamma, beta = self.film_gamma(study), self.film_beta(study)
        img = img * (1 + gamma) + beta

        x = torch.cat([self.img_proj(img), self.study_proj(study)], dim=1)
        logits = self.regressor(x).squeeze(1)

        loss = None
        if labels is not None:
            loss = nn.SmoothL1Loss()(logits, labels)  # try Huber; swap for MSE to compare
        return {"loss": loss, "logits": logits}