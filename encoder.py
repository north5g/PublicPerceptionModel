from transformers import CLIPModel, SiglipModel, AutoImageProcessor, CLIPProcessor, AutoModel, AutoProcessor, Blip2Model, GitModel

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

    elif name == "dinov2":
        model_id = "facebook/dinov2-base"
        encoder = AutoModel.from_pretrained(model_id)
        processor = AutoImageProcessor.from_pretrained(model_id)
        encoder_dim = 768
        image_size = 224

    elif name == "blip2":
        model_id = "Salesforce/blip2-opt-2.7b"
        encoder = Blip2Model.from_pretrained(model_id).vision_model
        processor = AutoProcessor.from_pretrained(model_id)
        encoder_dim = 1408
        image_size = 224
        
    # elif name == "fuyu":
    #     model_id = "adept/fuyu-8b"
    #     encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True).vision_tower
    #     processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    #     encoder_dim = 1024
    #     image_size = 224

    else:
        raise ValueError(f"Unknown encoder: {name}")

    return encoder, processor, encoder_dim, image_size
