import json
import pandas as pd
from sklearn.model_selection import train_test_split

import re
import os 

def clean_disease_name(name):
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

        # From symptoms only take only the ones with '-
        bullet_symptoms = []
        if symptoms_raw and symptoms_raw != "N/A":
            lines = symptoms_raw.split("\n")
            bullet_symptoms = [line.strip() for line in lines if line.strip().startswith("-")]

        cleaned_symptoms = " ".join(bullet_symptoms)

        # Combine input (in case we dont have symptoms or description)
        if description and cleaned_symptoms:
            input_text = f"{description} Symptoms: {cleaned_symptoms}"
        elif description:
            input_text = description
        elif cleaned_symptoms:
            input_text = f"Symptoms: {cleaned_symptoms}"
        else:
            continue 

        updated_data.append({
            "disease": disease,
            "input": input_text,
            "symptoms": cleaned_symptoms
        })

    return pd.DataFrame(updated_data)

dataset = load_and_preprocess_dataset(path="./data/boli_nhs.json")

if not os.path.exists("outputs"):
    os.makedirs("outputs")

dataset.to_csv("outputs/processed_dataset.csv", index=False)

