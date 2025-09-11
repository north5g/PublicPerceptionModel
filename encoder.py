from transformers import CLIPModel, SiglipModel, AutoImageProcessor, CLIPProcessor, AutoModel, AutoProcessor, Blip2Model, GitModel

# encoder_helpers.py
import torch

def resolve_feature_extractor(processor):
    # Some HF classes wrap a feature_extractor inside processor
    if processor is None:
        return None
    return getattr(processor, "feature_extractor", processor)

def get_image_size_from_processor(processor, fallback=224):
    fe = resolve_feature_extractor(processor)
    if fe is None:
        return fallback

    # common shapes: fe.size (dict or int) or fe.image_size or fe.size_or_max_length
    size = None
    if hasattr(fe, "size"):
        size = fe.size
    elif hasattr(fe, "image_size"):
        size = fe.image_size
    elif hasattr(fe, "size_or_max_length"):
        size = fe.size_or_max_length

    if size is None:
        return fallback

    # size might be dict {"height":.., "width":..} or int or tuple
    if isinstance(size, dict):
        return int(size.get("height") or size.get("width") or next(iter(size.values())))
    if isinstance(size, (list, tuple)):
        return int(size[0])
    return int(size)


def infer_encoder_dim_from_config(encoder):
    cfg = getattr(encoder, "config", None)
    if cfg is None:
        return None
    # common config attributes to check
    for attr in ("hidden_size", "projection_dim", "embed_dim", "dim", "image_embed_dim"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    # some models have a nested vision_config
    if hasattr(cfg, "vision_config"):
        vc = cfg.vision_config
        for attr in ("hidden_size", "projection_dim", "embed_dim", "dim"):
            if hasattr(vc, attr):
                return getattr(vc, attr)
    return None


def infer_encoder_dim_by_running_dummy(encoder, image_size, device=None):
    # last-resort: run a tiny forward to read shapes. Use CPU if no device provided
    device = device or torch.device("cpu")
    encoder = encoder.to(device)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    encoder.eval()
    with torch.no_grad():
        out = None
        try:
            out = encoder(pixel_values=dummy)
        except TypeError:
            out = encoder(dummy)
    # out can be dict or tensor
    if isinstance(out, dict):
        if "pooler_output" in out:
            return out["pooler_output"].shape[-1]
        if "image_embeds" in out:
            return out["image_embeds"].shape[-1]
        if "last_hidden_state" in out:
            return out["last_hidden_state"].shape[-1]
    elif torch.is_tensor(out):
        # for tensors, prefer the last dim
        return out.shape[-1]
    raise RuntimeError("Couldn't infer encoder dim from a dummy forward.")

def load_encoder(name: str):
    if name == "openclip":
        model_id = "openai/clip-vit-large-patch14"
        encoder = CLIPModel.from_pretrained(model_id).vision_model
        processor = CLIPProcessor.from_pretrained(model_id).feature_extractor

    elif name == "siglip":
        model_id = "google/siglip-so400m-patch14-384"
        encoder = SiglipModel.from_pretrained(model_id).vision_model
        processor = AutoImageProcessor.from_pretrained(model_id)

    elif name == "streetclip":
        model_id = "geolocal/StreetCLIP"
        encoder = CLIPModel.from_pretrained(model_id).vision_model
        processor = CLIPProcessor.from_pretrained(model_id).feature_extractor

    elif name == "dinov2":
        model_id = "facebook/dinov2-base"
        encoder = AutoModel.from_pretrained(model_id)
        processor = AutoImageProcessor.from_pretrained(model_id)

    elif name == "blip2":
        model_id = "Salesforce/blip2-opt-2.7b"
        encoder = Blip2Model.from_pretrained(model_id).vision_model
        processor = AutoProcessor.from_pretrained(model_id)
        
    # elif name == "fuyu":
    #     model_id = "adept/fuyu-8b"
    #     encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True).vision_tower
    #     processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown encoder: {name}")
    
     # Resolve image_size from processor
    image_size = get_image_size_from_processor(processor, fallback=224)

    # try config first
    encoder_dim = infer_encoder_dim_from_config(encoder)
    if encoder_dim is None:
        # fallback to a safe dummy forward on CPU
        encoder_dim = infer_encoder_dim_by_running_dummy(encoder, image_size, device=torch.device("cpu"))

    return encoder, processor, encoder_dim, image_size
