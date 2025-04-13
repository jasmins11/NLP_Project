import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import re

def clean_disease_name(name):
    # Elimină newline, "-", "Overview", spații multiple
    name = re.sub(r"Overview", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[\n\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def load_and_preprocess_dataset(path="./data/boli_nhs.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    updated_data = []
    for item in raw_data:
        disease = clean_disease_name(item.get("disease", "N/A"))
        description = item.get("description", "").strip()
        symptoms_raw = item.get("symptoms", "").strip()

        # Prelucrează simptomele (extrage doar liniile cu "-")
        bullet_symptoms = []
        if symptoms_raw and symptoms_raw != "N/A":
            lines = symptoms_raw.split("\n")
            bullet_symptoms = [line.strip() for line in lines if line.strip().startswith("-")]

        cleaned_symptoms = " ".join(bullet_symptoms)

        # Combină input-ul în funcție de ce avem
        if description and cleaned_symptoms:
            input_text = f"{description} Symptoms: {cleaned_symptoms}"
        elif description:
            input_text = description
        elif cleaned_symptoms:
            input_text = f"Symptoms: {cleaned_symptoms}"
        else:
            continue  # ignorăm dacă nu avem niciun text util

        updated_data.append({
            "disease": disease,
            "input": input_text,
            "symptoms": cleaned_symptoms
        })

    return pd.DataFrame(updated_data)

# Încărcăm datasetul
dataset = load_and_preprocess_dataset(path="./data/boli_nhs.json")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenizare text
def preprocess_function(examples):
    return tokenizer(examples, padding=True, truncation=True)

tokenized_dataset = dataset["input"].apply(preprocess_function)

# Split train/test
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Preview
print(train_data.head())
