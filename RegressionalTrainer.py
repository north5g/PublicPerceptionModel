# regressionalTrainer.py
import torch
from torch import nn
from typing import Optional, Tuple, Any
from encoder import get_image_size_from_processor  # adjust import to your project layout


class VisionTextRegressor(nn.Module):
    """
    VisionTextRegressor
    - encoder: a HuggingFace / torchvision-style encoder that accepts `pixel_values=...`.
    - num_study_types: size of the categorical study type embedding.
    - device: optional torch.device. If provided, encoder is moved to device early to reduce CPU memory.
    - freeze: if True the encoder is frozen and called under torch.no_grad(); otherwise it's trainable.
    - l1_lambda: optional L1 coefficient applied to regressor parameters (Lasso-like).
    - label_mean/label_std: optional floats for label normalization used only for computing the loss.
      (logits returned are in original scale; normalization is internal to the loss calculation).
    - smooth_l1_beta: beta param for SmoothL1Loss (Huber-like). Default 1.0.
    """
    def __init__(
        self,
        encoder: nn.Module,
        num_study_types: int,
        study_embed_dim: int = 128,
        encoder_dim: Optional[int] = None,
        processor: Any = None,
        image_size: Optional[int] = None,
        freeze: bool = True,
        l1_lambda: float = 0.0,
        label_mean: Optional[float] = None,
        label_std: Optional[float] = None,
        smooth_l1_beta: float = 1.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim  # inferred later if None
        self.l1_lambda = float(l1_lambda or 0.0)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.freeze = bool(freeze)
        self.image_size = int(image_size) if image_size is not None else None
        self.processor = processor
        self.image_size = image_size
        self.l1_lambda = float(l1_lambda or 0.0)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.label_mean = None if label_mean is None else float(label_mean)
        self.label_std = None if label_std is None else float(label_std)
        self.freeze = bool(freeze)

        # Move encoder to device early if provided (reduces CPU memory pressure)
        if device is not None:
            self.encoder.to(device)

        # Freeze encoder parameters if requested
        if self.freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # set eval to disable dropout/batchnorm behaviours
            try:
                self.encoder.eval()
            except Exception:
                pass
        else:
            # Optionally unfreeze only last N blocks if model naming uses `encoder.layer.X`
            # Keep default behavior: leave everything trainable if freeze=False.
            for name, p in self.encoder.named_parameters():
                # Example heuristic to unfreeze only last layers if you want:
                # if "encoder.layer.11" in name or "encoder.layer.10" in name:
                #     p.requires_grad = True
                # else:
                #     p.requires_grad = False
                pass

        # Infer encoder_dim by running a small forward on the same device as the encoder parameters.
        if encoder_dim is None:
            img_size = self.image_size or get_image_size_from_processor(self.processor, fallback=224)
            encoder_device = next(self.encoder.parameters()).device
            dummy = torch.randn(1, 3, img_size, img_size, device=encoder_device)
            # We run a forward in no_grad to infer sizes (safe even if encoder is trainable).
            with torch.no_grad():
                # Many HF models accept pixel_values keyword
                try:
                    out = self.encoder(pixel_values=dummy)
                except TypeError:
                    out = self.encoder(dummy)

            # parse known HF output formats
            if isinstance(out, dict):
                if "pooler_output" in out:
                    img = out["pooler_output"]
                elif "last_hidden_state" in out:
                    img = out["last_hidden_state"][:, 0]
                elif "image_embeds" in out:
                    img = out["image_embeds"]
                else:
                    # Last resort: pick first tensor-like value
                    first_val = next(iter(out.values()))
                    if hasattr(first_val, "shape"):
                        if first_val.ndim == 3:
                            img = first_val[:, 0]
                        else:
                            img = first_val
                    else:
                        raise ValueError(f"Unknown encoder output shape/keys: {out.keys()}")
            else:
                # out might be a tensor or tuple
                if hasattr(out, "shape"):
                    if out.ndim == 3:
                        img = out[:, 0]
                    else:
                        img = out
                elif isinstance(out, (list, tuple)):
                    candidate = out[0]
                    if candidate.ndim == 3:
                        img = candidate[:, 0]
                    else:
                        img = candidate
                else:
                    raise ValueError("Unable to infer encoder output format.")

            encoder_dim = int(img.shape[1])

        self.encoder_dim = encoder_dim

        # Study embeddings + FiLM layers
        self.study_embedding = nn.Embedding(num_study_types, study_embed_dim)
        self.film_gamma = nn.Linear(study_embed_dim, encoder_dim)
        self.film_beta  = nn.Linear(study_embed_dim, encoder_dim)

        # Projections for encoder output and study embedding
        self.img_proj   = nn.Sequential(
            nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, 512), nn.ReLU()
        )
        self.study_proj = nn.Sequential(
            nn.LayerNorm(study_embed_dim), nn.Linear(study_embed_dim, 128), nn.ReLU()
        )

        # Regressor head
        self.regressor  = nn.Sequential(
            nn.Linear(512 + 128, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Initialize regressor weights for stability
        self._init_weights()

    def _init_weights(self):
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # small init for FiLM layers
        nn.init.xavier_uniform_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.xavier_uniform_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def to_device(self, device: torch.device):
        """
        Convenience: move model (and underlying encoder) to device.
        Use this if you didn't pass `device` at construction.
        """
        self.to(device)
        try:
            self.encoder.to(device)
        except Exception:
            pass
        return self

    def set_label_stats(self, mean: float, std: float):
        """Set label mean/std for internal normalization used in loss computation."""
        self.label_mean = float(mean)
        self.label_std = float(std)

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Returns the encoder image embedding (detached, CPU tensor) for a batch of pixel_values.
        Useful to extract features for Ridge/Lasso baselines.
        """
        device = next(self.encoder.parameters()).device
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            try:
                out = self.encoder(pixel_values=pixel_values)
            except TypeError:
                out = self.encoder(pixel_values)
            if isinstance(out, dict):
                if "pooler_output" in out:
                    img = out["pooler_output"]
                elif "last_hidden_state" in out:
                    img = out["last_hidden_state"][:, 0]
                elif "image_embeds" in out:
                    img = out["image_embeds"]
                else:
                    # pick first tensor-like
                    first_val = next(iter(out.values()))
                    img = first_val[:, 0] if first_val.ndim == 3 else first_val
            else:
                if hasattr(out, "ndim") and out.ndim == 3:
                    img = out[:, 0]
                elif isinstance(out, (list, tuple)):
                    img = out[0]
                else:
                    img = out

            # Return detached CPU tensor
            return img.detach().cpu()

    def forward(self, pixel_values: torch.Tensor, study_type_ids: torch.LongTensor,
                labels: Optional[torch.FloatTensor] = None) -> dict:
        """
        Forward pass:
        - If self.freeze is True, encoder is called under torch.no_grad() (no grads for encoder).
        - Otherwise, encoder receives gradients if it has requires_grad True.
        - Loss uses SmoothL1Loss (Huber-like) and optional L1 penalty on regressor weights.
        - If label_mean & label_std are set, the loss is computed on normalized labels/predictions
          to improve numerical stability. The returned logits remain in the model's original scale.
        """
        # Ensure tensors are on the same device as encoder parameters (Trainer does this for you as well,
        # but this makes the model robust to being called standalone).
        encoder_device = next(self.encoder.parameters()).device
        pixel_values = pixel_values.to(encoder_device)
        study_type_ids = study_type_ids.to(encoder_device)
        if labels is not None:
            labels = labels.to(encoder_device)

        # Run encoder (conditionally no_grad if frozen)
        if self.freeze:
            with torch.no_grad():
                try:
                    out = self.encoder(pixel_values=pixel_values)
                except TypeError:
                    out = self.encoder(pixel_values)
        else:
            try:
                out = self.encoder(pixel_values=pixel_values)
            except TypeError:
                out = self.encoder(pixel_values)

        # Parse encoder output to get a [B, encoder_dim] image embedding
        if isinstance(out, dict):
            if "pooler_output" in out:
                img = out["pooler_output"]
            elif "last_hidden_state" in out:
                img = out["last_hidden_state"][:, 0]
            elif "image_embeds" in out:
                img = out["image_embeds"]
            else:
                # fallback: first tensor-like value
                first_val = next(iter(out.values()))
                img = first_val[:, 0] if first_val.ndim == 3 else first_val
        else:
            if hasattr(out, "ndim") and out.ndim == 3:
                img = out[:, 0]
            elif isinstance(out, (list, tuple)):
                img = out[0]
            else:
                img = next(iter(outputs.values()))
        else:
            img = outputs

        # FiLM conditioning with study embeddings
        study = self.study_embedding(study_type_ids)
        gamma = torch.tanh(self.film_gamma(study))
        beta = self.film_beta(study)
        img = img * (1.0 + gamma) + beta

        # combine projections and predict
        x = torch.cat([self.img_proj(img), self.study_proj(study)], dim=1)
        logits = self.regressor(x).squeeze(1)


        loss = None
        if labels is not None:
            # Prepare normalized versions if requested (stable training)
            if (self.label_mean is not None) and (self.label_std is not None) and (self.label_std != 0.0):
                labels_norm = (labels - self.label_mean) / (self.label_std + 1e-12)
                preds_norm = (logits - self.label_mean) / (self.label_std + 1e-12)
            else:
                labels_norm = labels
                preds_norm = logits

            loss_fn = nn.SmoothL1Loss(beta=self.smooth_l1_beta, reduction="mean")
            regression_loss = loss_fn(preds_norm, labels_norm)

            if self.l1_lambda and self.l1_lambda > 0.0:
                l1_norm = torch.tensor(0.0, device=regression_loss.device)
                for p in self.regressor.parameters():
                    if p.requires_grad:
                        l1_norm = l1_norm + p.abs().sum()
                loss = regression_loss + self.l1_lambda * l1_norm
            else:
                loss = regression_loss

        return {"loss": loss, "logits": logits}
