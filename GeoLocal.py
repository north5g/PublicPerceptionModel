import torch
import random
import numpy as np
from unsloth import FastVisionModel
from transformers import AutoImageProcessor, TrainingArguments, Trainer
from data_prep_text import PlacePulseDatasetText, labels  # <-- Use the text dataset
from torchvision import transforms

# 1. Set random seed for reproducibility
def set_seed(seed=3407):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(3407)

# 2. Define transforms (with normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load model and tokenizer
model_name = "Salesforce/blip2-opt-2.7b"
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    use_gradient_checkpointing=True,
    device_map="auto"
)
# For BLIP-2, use the language model as text encoder
text_encoder = model.language_model

image_processor = AutoImageProcessor.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 4. Wrap model with LoRA/PEFT
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=False,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 5. Prepare datasets using PlacePulseDatasetText
votes_path = "votes.tsv"
train_dataset = PlacePulseDatasetText(
    votes_tsv_path=votes_path,
    transform=transform,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    device=device,
    split='train',
)
val_dataset = PlacePulseDatasetText(
    votes_tsv_path=votes_path,
    transform=transform,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    device=device,
    split='val',
)

# 6. Custom data collator for batching label embeddings
def data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    return {"pixel_values": pixel_values, "labels": labels}

# 7. Subclass model for cosine similarity loss
import torch.nn as nn
import torch.nn.functional as F

class PlacePulseVisionModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, pixel_values, labels=None):
        # Get image embeddings
        image_embeds = self.base_model.vision_model(pixel_values).last_hidden_state.mean(dim=1)
        loss = None
        if labels is not None:
            loss = 1 - F.cosine_similarity(image_embeds, labels).mean()
        return {"loss": loss, "logits": image_embeds}

model = PlacePulseVisionModel(model).to(device)

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    remove_unused_columns=False,
    seed=3407,
    save_total_limit=2,
)

# 9. Trainer setup
from placepulse_accuracy import compute_placepulse_pairwise_accuracy

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,
    data_collator=data_collator,
    compute_metrics=lambda eval_pred: compute_placepulse_pairwise_accuracy(
        eval_pred, val_dataset, labels, device
    ),
)

trainer.train()
