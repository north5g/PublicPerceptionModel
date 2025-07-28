from transformers import CLIPModel, SiglipModel, AutoImageProcessor, CLIPProcessor, AutoModel, AutoProcessor

def load_encoder(name: str):
    if name == "openclip":
        model_id = "openai/clip-vit-large-patch14"
        encoder = CLIPModel.from_pretrained(model_id).vision_model
        processor = CLIPProcessor.from_pretrained(model_id).feature_extractor
        encoder_dim = 768
        image_size = 224

    elif name == "siglip":
        model_id = "google/siglip-so400m-patch14-384"
        encoder = SiglipModel.from_pretrained(model_id).vision_model
        processor = AutoImageProcessor.from_pretrained(model_id)
        encoder_dim = 768
        image_size = 384

    elif name == "streetclip":
        model_id = "geolocal/StreetCLIP"
        encoder = CLIPModel.from_pretrained(model_id).vision_model
        processor = CLIPProcessor.from_pretrained(model_id).feature_extractor
        encoder_dim = 768
        image_size = 336

    elif name == "qwen":
        model_id = "Qwen/Qwen-VL"
        encoder = AutoModel.from_pretrained(model_id).vision_tower  # typical for vision-language models
        processor = AutoProcessor.from_pretrained(model_id)
        encoder_dim = encoder.config.hidden_size  # safer than hardcoding
        image_size = 448  # typical for Qwen-VL

    else:
        raise ValueError(f"Unknown encoder: {name}")

    return encoder, processor, encoder_dim, image_size
