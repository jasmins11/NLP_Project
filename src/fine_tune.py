import time
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Configurare
MODEL_NAME = "distilgpt2"
BATCH_SIZE = 8
EPOCHS = 4
MAX_LENGTH = 128

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tokenizer și model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)


# Dataset personalizat
class LanguageDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe["input"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encodings = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()  # GPT2: input == output (shifted)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# Încarcă datasetul preprocesat
df = pd.read_csv("outputs/processed_dataset.csv")  # sau încarcă direct din scriptul tău
dataset = LanguageDataset(df, tokenizer)

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# Loss & optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Training
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()

    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in train_loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    duration = time.time() - start_time
    print(f"Epoch {epoch+1} completed in {duration:.2f}s | Train Loss: {avg_train_loss:.4f}")

# Salvare
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_tokenizer")
print("Modelul și tokenizer-ul au fost salvate.")