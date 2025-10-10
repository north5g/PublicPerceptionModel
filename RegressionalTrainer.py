import torch
from typing import Optional, Any, Iterable, List, Tuple
from torch import nn
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback
import logging

logger = logging.getLogger(__name__)


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
        encoder_dim: int,
        study_embed_dim: int = 128,
        freeze: bool = True,
        l1_lambda: float = 0.0,
        smooth_l1_beta: float = 1.0,
        image_size: Optional[int] = None, # <--- added to accept callers passing this kwarg
        processor: Optional[Any] = None, # optional image processor (keeps compatibility)
        device: Optional[torch.device] = None,
        label_mean: Optional[float] = None,
        label_std: Optional[float] = None,
        loss_type: str = "huber"  # or "mse"
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim  # inferred later if None
        self.l1_lambda = float(l1_lambda or 0.0)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.freeze = bool(freeze)
        self.image_size = int(image_size) if image_size is not None else None
        self.processor = processor
        self.label_mean = float(label_mean) if label_mean is not None else None
        self.label_std = float(label_std) if label_std is not None else None
        self.loss_type = loss_type.lower()
        assert self.loss_type in ("huber", "mse", "mae"), "loss_type must be 'huber', 'mse', or 'mae'"

        if device is not None:
            try:
                self.encoder.to(device)
            except Exception:
                pass


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

    def forward(self, pixel_values: torch.Tensor, study_type_ids: torch.LongTensor,
                labels: torch.FloatTensor = None) -> dict:
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
                outputs = self.encoder(pixel_values=pixel_values)
        else:
            outputs = self.encoder(pixel_values=pixel_values)

        if isinstance(outputs, dict):
            if "pooler_output" in outputs:
                img = outputs["pooler_output"]
            elif "last_hidden_state" in outputs:
                img = outputs["last_hidden_state"][:, 0]
            elif "image_embeds" in outputs:
                img = outputs["image_embeds"]
            else:
                img = next(iter(outputs.values()))
        else:
            img = outputs

        # FiLM conditioning
        study = self.study_embedding(study_type_ids)
        gamma = torch.tanh(self.film_gamma(study))
        beta = self.film_beta(study)
        img = img * (1.0 + gamma) + beta


        # Combine projections and regress
        x = torch.cat([self.img_proj(img), self.study_proj(study)], dim=1)
        logits = self.regressor(x).squeeze(1)


        loss = None
        if labels is not None:
            # make shapes consistent
            if logits.shape != labels.shape:
                try:
                    labels = labels.view_as(logits)
                except Exception:
                    labels = labels.reshape(logits.shape)

            # normalize if requested
            if (self.label_mean is not None) and (self.label_std is not None):
                lm = torch.tensor(self.label_mean, device=logits.device, dtype=logits.dtype)
                ls = torch.tensor(self.label_std, device=logits.device, dtype=logits.dtype)
                logits_norm = (logits - lm) / (ls + 1e-12)
                labels_norm = (labels - lm) / (ls + 1e-12)
            else:
                logits_norm = logits
                labels_norm = labels

            if self.loss_type == "mse":
                base_loss = F.mse_loss(logits_norm, labels_norm)
            elif self.loss_type == "mae":
                base_loss = F.l1_loss(logits_norm, labels_norm)
            else:  # huber / smooth l1
                # torch.nn.functional.smooth_l1_loss uses 'beta' argument
                base_loss = F.smooth_l1_loss(logits_norm, labels_norm, beta=self.smooth_l1_beta, reduction="mean")

            # L1 regularization limited to regressor head by default
            if self.l1_lambda > 0.0:
                l1_norm = base_loss.new_zeros(())
                for p in self.regressor.parameters():
                    if p.requires_grad:
                        l1_norm = l1_norm + p.abs().sum()
                loss = base_loss + self.l1_lambda * l1_norm
            else:
                loss = base_loss

        return {"loss": loss, "logits": logits}




class L1Trainer(Trainer):
    """
    Trainer subclass that:
    - Optionally adds L1 regularization (applied to the regressor head by default).
    - Allows specifying a loss_type fallback if model doesn't return a precomputed loss.
    Usage: pass l1_lambda and loss_type when constructing this Trainer.
    """
    def __init__(self, l1_lambda: float = 0.0, l1_scope: str = "regressor",
                 loss_type: Optional[str] = None, *args, **kwargs):
        """
        l1_lambda: coefficient for L1 penalty.
        l1_scope: "regressor" | "head" | "all" -- which parameters to include for L1.
        loss_type: if provided, overrides model.loss_type when the model returns no loss.
        """
        super().__init__(*args, **kwargs)
        self.l1_lambda = float(l1_lambda or 0.0)
        assert l1_scope in ("regressor", "head", "all")
        self.l1_scope = l1_scope
        self.loss_type = loss_type

    def _iter_l1_params(self, model) -> Iterable[nn.Parameter]:
        if self.l1_scope == "regressor" and hasattr(model, "regressor"):
            return model.regressor.parameters()
        elif self.l1_scope == "head":
            parts = []
            for name in ("regressor", "study_proj", "img_proj", "film_gamma", "film_beta", "study_embedding"):
                if hasattr(model, name):
                    obj = getattr(model, name)
                    if isinstance(obj, nn.Module):
                        parts.extend(list(obj.parameters()))
            return iter(parts)
        else:
            return model.parameters()

    def compute_loss(self, model, inputs, return_outputs: bool = False, *args, **kwargs):
        """
        Backwards-compatible compute_loss that tolerates extra kwargs passed by different
        Trainer versions (e.g. num_items_in_batch). Old behaviour preserved.

        Args:
            model: the model being trained
            inputs: data batch (expected to contain 'labels' and features)
            return_outputs: if True return (loss, outputs)
            *args, **kwargs: accept unexpected Trainer arguments (ignored unless used)
        """
        # steal optional Trainer-supplied args if present (no-op otherwise)
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        # if there are any other unexpected kwargs, keep them for debugging
        extra_kwargs = {k: v for k, v in kwargs.items()}

        # Let the model produce outputs (and possibly a loss)
        outputs = model(**inputs)
        loss = outputs.get("loss", None)
        logits = outputs.get("logits", None)

        # Fallback: if model didn't compute loss, compute it here using labels
        if loss is None:
            if logits is None:
                raise ValueError("Model didn't return logits; cannot compute fallback loss.")
            labels = inputs.get("labels", None)
            if labels is None:
                raise ValueError("No labels found in inputs to compute loss.")
            labels = labels.to(logits.device).float()

            # Respect label normalization if model has mean/std attributes
            mean = getattr(model, "label_mean", None)
            std = getattr(model, "label_std", None)
            if (mean is not None) and (std is not None):
                lm = torch.tensor(mean, device=logits.device, dtype=logits.dtype)
                ls = torch.tensor(std, device=logits.device, dtype=logits.dtype)
                logits_norm = (logits - lm) / (ls + 1e-12)
                labels_norm = (labels - lm) / (ls + 1e-12)
            else:
                logits_norm = logits
                labels_norm = labels

            use_loss_type = self.loss_type or getattr(model, "loss_type", "huber")
            if use_loss_type == "mse":
                loss = F.mse_loss(logits_norm, labels_norm)
            elif use_loss_type == "mae":
                loss = F.l1_loss(logits_norm, labels_norm)
            else:
                beta = getattr(model, "smooth_l1_beta", 1.0)
                loss = F.smooth_l1_loss(logits_norm, labels_norm, beta=beta)

        # Add L1 regularization on selected parameter subset if requested
        if self.l1_lambda > 0.0:
            l1_norm = loss.new_zeros(())
            for p in self._iter_l1_params(model):
                if p.requires_grad:
                    l1_norm = l1_norm + p.abs().sum()
            loss = loss + self.l1_lambda * l1_norm

        # Optional: log unexpected kwargs (non-fatal)
        if extra_kwargs:
            logger.debug(f"compute_loss received extra kwargs: {list(extra_kwargs.keys())}")

        return (loss, outputs) if return_outputs else loss



class GradNormCallback(TrainerCallback):
    """
    Callback that logs gradient norms and maximum absolute gradient after each step.
    Helps identify which step/layers produce exploding gradients.
    - threshold: only print when grad_norm > threshold (avoid spam).
    """
    def __init__(self, threshold: float = 50.0, log_fn: Optional[Any] = None):
        self.threshold = float(threshold)
        self.log_fn = log_fn or (lambda *a, **k: logger.warning(*a, **k))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # model may be None in some contexts; guard
        if model is None:
            return
        total_sq = 0.0
        max_abs = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                gnorm = float(g.norm(2).item())
                total_sq += gnorm ** 2
                max_abs = max(max_abs, float(g.abs().max().item()))
        grad_norm = total_sq ** 0.5
        if grad_norm > self.threshold:
            # print a single-line alert; users can hook logger to capture this
            self.log_fn(f"[GradNormCallback] step={state.global_step} grad_norm={grad_norm:.2f} max_grad_abs={max_abs:.3e}")


# Utility helper for building param groups (encoder vs head)
def get_param_groups(model: nn.Module, encoder_lr: float, head_lr: float,
                     weight_decay: float = 0.01) -> List[dict]:
    """
    Returns param-groups suitable for passing to torch.optim.AdamW.
    Heuristic: parameters whose name contains 'encoder' use encoder_lr; others use head_lr.
    You can adapt the filter to your exact encoder naming convention.
    """
    encoder_params = []
    head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in n or "vision" in n or "pixel" in n:
            encoder_params.append(p)
        else:
            head_params.append(p)

    groups = [
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": head_params,    "lr": head_lr,    "weight_decay": 0.0}
    ]
    return groups