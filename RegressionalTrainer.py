from torch import nn
from typing import Optional
import torch
from transformers import Trainer, TrainingArguments

class VisionTextRegressor(nn.Module):
    def __init__(self, encoder, num_study_types, study_embed_dim=128, image_size = 336):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Dummy pass to infer encoder output dimension
        dummy_input = torch.randn(1, 3, image_size, image_size)  # Standard input shape
        with torch.no_grad():
            encoder_output = self.encoder(dummy_input.to(next(self.encoder.parameters()).device))
            if isinstance(encoder_output, dict):
                # Handle transformer-style outputs
                if "last_hidden_state" in encoder_output:
                    image_features = encoder_output["last_hidden_state"].mean(dim=1)
                elif "pooler_output" in encoder_output:
                    image_features = encoder_output["pooler_output"]
                elif "image_embeds" in encoder_output:
                    image_features = encoder_output["image_embeds"]
                else:
                    raise ValueError("Unknown encoder output format")
            else:
                image_features = encoder_output  # Assume it's already a tensor
        encoder_dim = image_features.shape[1]

        self.study_embedding = nn.Embedding(num_study_types, study_embed_dim)

        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim + study_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, pixel_values, study_type_ids, labels: Optional[torch.FloatTensor] = None):
        with torch.no_grad():
            encoder_output = self.encoder(pixel_values)
            if isinstance(encoder_output, dict):
                if "last_hidden_state" in encoder_output:
                    image_features = encoder_output["last_hidden_state"].mean(dim=1)
                elif "pooler_output" in encoder_output:
                    image_features = encoder_output["pooler_output"]
                elif "image_embeds" in encoder_output:
                    image_features = encoder_output["image_embeds"]
                else:
                    raise ValueError("Unknown encoder output format")
            else:
                image_features = encoder_output

        study_embeddings = self.study_embedding(study_type_ids)
        combined = torch.cat([image_features, study_embeddings], dim=1)

        logits = self.regressor(combined).squeeze(1)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(logits, labels)

        return {"loss": loss, "logits": logits}

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(
            pixel_values=inputs["pixel_values"],
            study_type_ids=inputs["study_type_ids"]
        )
        loss = nn.MSELoss()(outputs, labels)
        return (loss, outputs) if return_outputs else loss