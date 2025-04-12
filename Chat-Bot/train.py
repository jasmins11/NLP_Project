import pandas as pd
import json
import time
from tqdm import tqdm

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split


with open("./WebScraping/boli_nhs.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

updated_data = []
for item in raw_data:
    name = item.get("boala", "N/A")
    descriere = item.get("descriere", "N/A")
    simptome = item.get("simptome", "N/A")
    updated_data.append({
        "input": f"Diseases: {name} | Description: {descriere} | Symptoms: {simptome}"
    })

df = pd.DataFrame(updated_data)

print(df.head(2))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(device)


class LanguageDataset(TorchDataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe.to_dict(orient='records')
        self.tokenizer = tokenizer
        self.max_length = self._calculate_max_length(dataframe)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["input"]
        tokens = self.tokenizer.encode_plus(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        return tokens

    def _calculate_max_length(self, df):
        max_len = max(df["input"].apply(len))
        x = 2
        while x < max_len:
            x *= 2
        return min(x, 1024)

dataset = LanguageDataset(df, tokenizer)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

num_epochs = 4
model_name = "distilgpt2"
gpu = 0
batch_size = 8

results = pd.DataFrame(columns=['epoch', 'transformer', 'batch_size', 'gpu', 'training_loss', 'validation_loss', 'epoch_duration_sec'])

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    epoch_training_loss = 0
    train_iterator = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{num_epochs}")

    for batch in train_iterator:
        optimizer.zero_grad()
        inputs = batch['input_ids'].squeeze(1).to(device)
        targets = inputs.clone()
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_training_loss += loss.item()
        train_iterator.set_postfix({'Training Loss': loss.item()})

    avg_train_loss = epoch_training_loss / len(train_loader)

    model.eval()
    epoch_validation_loss = 0
    val_iterator = tqdm(val_loader, desc=f"Validating epoch {epoch + 1}/{num_epochs}")

    with torch.no_grad():
        for batch in val_iterator:
            inputs = batch['input_ids'].squeeze(1).to(device)
            targets = inputs.clone()
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            epoch_validation_loss += loss.item()
            val_iterator.set_postfix({'Validation Loss': loss.item()})

    avg_val_loss = epoch_validation_loss / len(val_loader)
    epoch_duration = time.time() - start_time

    results.loc[len(results)] = {
        'epoch': epoch + 1,
        'transformer': model_name,
        'batch_size': batch_size,
        'gpu': gpu,
        'training_loss': avg_train_loss,
        'validation_loss': avg_val_loss,
        'epoch_duration_sec': epoch_duration
    }

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Time = {epoch_duration:.2f}s")


model_path = "trained_model"
tokenizer_path = "trained_tokenizer"

model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

print(f" Model saved in '{model_path}'")
print(f" Tokenizer saved in : '{tokenizer_path}'")
