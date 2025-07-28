from transformers import TrainingArguments
from PlacePulseDataset import PlacePulseDataset
from encoder import load_encoder
from RegressionalTrainer import VisionTextRegressor
from torchvision import transforms
from transformers import Trainer

models = ["openclip",
         "siglip",
         "streetclip", 
         "qwen"]

# actual training set
# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     learning_rate=5e-5,
#     num_train_epochs=10,
#     logging_dir="./logs",
#     logging_steps=20,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True
# )

# test training set
training_args = TrainingArguments(
    output_dir="./model_test_results",
    per_device_train_batch_size=4,        # small batch size for faster iterations
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=1,                   # just 1 epoch
    max_steps=10,                         # or limit steps directly
    logging_dir="./logs",
    logging_steps=1,
    eval_strategy="no",             # disable eval to save time
    save_strategy="no",                   # disable saving
    load_best_model_at_end=False          # no need to track best model
)

from sklearn.metrics import mean_squared_error, r2_score

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    return {
        "mse": mean_squared_error(labels, preds),
        "r2": r2_score(labels, preds)
    }

# 1. Load encoder
encoder, processor, encoder_dim, image_size = load_encoder(models[2])

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# 2. Init model
model = VisionTextRegressor(
    encoder=encoder,
    num_study_types=6,
    image_size=image_size
)
dataset = PlacePulseDataset(transform = transform)
train_dataset, eval_dataset = dataset.split()

from transformers import default_data_collator

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()