# Put this in the same file that currently contains your encoder helpers (or import it)
import os
import torch
from transformers import (
    CLIPModel, CLIPProcessor, SiglipModel, AutoImageProcessor,
    AutoModel, AutoProcessor, Blip2Model
)

def resolve_feature_extractor(processor):
    """Return the inner image processor/feature_extractor if present.
       Supports older 'feature_extractor' and newer 'image_processor' fields."""
    if processor is None:
        return None
    for attr in ("feature_extractor", "image_processor"):
        val = getattr(processor, attr, None)
        if val is not None:
            return val
    # fallback to the processor itself (some HF classes are the extractor)
    return processor


def get_image_size_from_processor(processor, fallback=224):
    fe = resolve_feature_extractor(processor)
    if fe is None:
        return fallback

    size = None
    if hasattr(fe, "size"):
        size = fe.size
    elif hasattr(fe, "image_size"):
        size = fe.image_size
    elif hasattr(fe, "size_or_max_length"):
        size = fe.size_or_max_length

    if size is None:
        return fallback

    if isinstance(size, dict):
        return int(size.get("height") or size.get("width") or next(iter(size.values())))
    if isinstance(size, (list, tuple)):
        return int(size[0])
    return int(size)


def infer_encoder_dim_from_config(encoder):
    cfg = getattr(encoder, "config", None)
    if cfg is None:
        return None
    for attr in ("hidden_size", "projection_dim", "embed_dim", "dim", "image_embed_dim"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    if hasattr(cfg, "vision_config"):
        vc = cfg.vision_config
        for attr in ("hidden_size", "projection_dim", "embed_dim", "dim"):
            if hasattr(vc, attr):
                return getattr(vc, attr)
    return None


def infer_encoder_dim_by_running_dummy(encoder, image_size, device=None):
    device = device or torch.device("cpu")
    encoder = encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        try:
            out = encoder(pixel_values=dummy)
        except TypeError:
            out = encoder(dummy)
    if isinstance(out, dict):
        if "pooler_output" in out:
            return out["pooler_output"].shape[-1]
        if "image_embeds" in out:
            return out["image_embeds"].shape[-1]
        if "last_hidden_state" in out:
            return out["last_hidden_state"].shape[-1]
    elif torch.is_tensor(out):
        return out.shape[-1]
    raise RuntimeError("Couldn't infer encoder dim from a dummy forward.")


def load_encoder(name: str, device: torch.device = None, use_device_map: bool = True, low_cpu_mem_usage: bool = True):
    """
    Robust loader:
      - If CUDA available, first try HF accelerated load: device_map='auto', low_cpu_mem_usage
      - If that fails (no accelerate or unsupported), fall back to CPU load then .to(device)
    Returns: encoder_module, processor, encoder_dim, image_size
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    use_cuda = device.type == "cuda"
    torch_dtype = torch.float16 if (use_cuda and torch.cuda.is_available()) else torch.float32

    def _hf_from_pretrained(cls, model_id):
        # try accelerated load if we have CUDA and device_map requested
        try:
            if use_cuda and use_device_map:
                return cls.from_pretrained(model_id,
                                           device_map="auto",
                                           low_cpu_mem_usage=low_cpu_mem_usage,
                                           torch_dtype=torch_dtype)
            else:
                # simpler: load normally, then move to device
                return cls.from_pretrained(model_id)
        except Exception as e:
            print(f"[warning] accelerated/auto device_map load failed for {model_id}: {e}")
            print("Falling back to CPU load then .to(device).")
            model = cls.from_pretrained(model_id)
            if use_cuda:
                model.to(device)
            return model

    if name == "openclip":
        model_id = "openai/clip-vit-large-patch14"
        model = _hf_from_pretrained(CLIPModel, model_id)
        encoder = getattr(model, "vision_model", model)
        processor = CLIPProcessor.from_pretrained(model_id)

    elif name == "siglip":
        model_id = "google/siglip-so400m-patch14-384"
        model = _hf_from_pretrained(SiglipModel, model_id)
        encoder = getattr(model, "vision_model", model)
        processor = AutoImageProcessor.from_pretrained(model_id)

    elif name == "streetclip":
        model_id = "geolocal/StreetCLIP"
        model = _hf_from_pretrained(CLIPModel, model_id)
        encoder = getattr(model, "vision_model", model)
        # use the HF processor (works across CLIP versions)
        processor = CLIPProcessor.from_pretrained(model_id)

    elif name == "dinov2":
        model_id = "facebook/dinov2-base"
        model = _hf_from_pretrained(AutoModel, model_id)
        encoder = model
        processor = AutoImageProcessor.from_pretrained(model_id)

    elif name == "blip2":
        model_id = "Salesforce/blip2-opt-2.7b"
        model = _hf_from_pretrained(Blip2Model, model_id)
        encoder = getattr(model, "vision_model", model)
        processor = AutoProcessor.from_pretrained(model_id)

    else:
        raise ValueError(f"Unknown encoder: {name}")

    # At this point, model may already be placed across devices (device_map) or entirely on device.
    # Ensure encoder is moved to target device if it's not using device_map.
    try:
        # if encoder has parameters placed on CPU but we want them on single GPU, move them
        if not any(p.device.type == "cuda" for p in encoder.parameters()) and use_cuda:
            encoder = encoder.to(device)
    except Exception:
        # encoder might be a sharded object under accelerate, skip moving in that case
        pass

    image_size = get_image_size_from_processor(processor, fallback=224)
    encoder_dim = infer_encoder_dim_from_config(encoder)
    if encoder_dim is None:
        # run a small dummy forward on the same device as the encoder (if device_map used, let HF route)
        encoder_dim = infer_encoder_dim_by_running_dummy(encoder, image_size, device=device)

    # final sanity prints to help debug — remove in production
    print(f"[load_encoder] model_id={model_id}, target_device={device}, encoder_dim={encoder_dim}, image_size={image_size}")
    try:
        print(f"[load_encoder] example parameter device: {next(encoder.parameters()).device}")
    except StopIteration:
        pass

    return encoder, processor, encoder_dim, image_size
