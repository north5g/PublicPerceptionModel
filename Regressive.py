import torch
import random
import numpy as np
from unsloth import FastVisionModel
from transformers import TrainingArguments
from data_prep_weights import PlacePulseDatasetWeight
from torchvision import transforms
from trl import SFTTrainer, SFTConfig
import torch.nn as nn

PlacePulseDatasetWeight.preprocess()
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

# 3. Study type mapping (should match your dataset)
STUDY_TYPES = [
    'safe', 'lively', 'clean', 'wealthy', 'depressing', 'beautiful'
]
STUDY_TYPE_TO_IDX = {name: i for i, name in enumerate(STUDY_TYPES)}

# 4. Load model and tokenizer
model_name = "Salesforce/blip2-opt-2.7b"
base_model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    use_gradient_checkpointing=True,
    device_map="auto"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 5. Optionally wrap model with LoRA/PEFT
base_model = FastVisionModel.get_peft_model(
    base_model,
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

# 6. Prepare datasets using PlacePulseDatasetWeight
class MultiTaskPlacePulseDatasetWeight(PlacePulseDatasetWeight):
    def __getitem__(self, idx):
        location_id = self.dataframe.iloc[idx]['location_id']
        img = self.get_img_by_location_id(location_id)
        weight = self.dataframe.iloc[idx]['weight']
        study_type = self.dataframe.iloc[idx]['study_type']
        study_type_idx = STUDY_TYPE_TO_IDX[study_type]
        if self.transform:
            img = self.transform(img)
        sample = {
            "pixel_values": img,
            "labels": weight,
            "study_type_idx": study_type_idx
        }
        if self.return_location_id:
            sample["location_id"] = location_id
        return sample

train_dataset = MultiTaskPlacePulseDatasetWeight(
    qscores_tsv_path="place-pulse-2.0/qscores.tsv",
    transform=transform,
    split='train',
)
val_dataset = MultiTaskPlacePulseDatasetWeight(
    qscores_tsv_path="place-pulse-2.0/qscores.tsv",
    transform=transform,
    split='val',
)

# 7. Multi-task regression model (with Unsloth compatibility)
class MultiTaskBlip2RegressionModel(nn.Module):
    def __init__(self, base_model, num_study_types):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.vision_model.config.hidden_size
        self.study_type_embed = nn.Embedding(num_study_types, hidden_size)
        self.regressor = nn.Linear(hidden_size * 2, 1)
        self.config = base_model.config

    def forward(self, pixel_values, study_type_idx=None, labels=None):
        vision_outputs = self.base_model.vision_model(pixel_values)
        img_feat = vision_outputs.last_hidden_state.mean(dim=1)
        study_feat = self.study_type_embed(study_type_idx)
        combined = torch.cat([img_feat, study_feat], dim=1)
        pred = self.regressor(combined).squeeze(-1)
        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(pred, labels.float())
        return {"loss": loss, "logits": pred}

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        # For Unsloth compatibility
        return self.regressor

model = MultiTaskBlip2RegressionModel(base_model, num_study_types=len(STUDY_TYPES)).to(device)

# 8. Data collator for Trainer
def data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
    study_type_idx = torch.tensor([f["study_type_idx"] for f in features], dtype=torch.long)
    batch = {
        "pixel_values": pixel_values,
        "labels": labels,
        "study_type_idx": study_type_idx
    }
    if "location_id" in features[0]:
        batch["location_id"] = [f["location_id"] for f in features]
    return batch

# 9. Training arguments (SFTConfig for Unsloth SFTTrainer)
train_args = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    seed=3407,
    save_total_limit=2,
    remove_unused_columns=False,
)

# 10. Trainer setup
trainer = SFTTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()