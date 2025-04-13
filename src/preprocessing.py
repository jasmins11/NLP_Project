import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Funcția pentru a încărca și preprocesa datasetul
def load_and_preprocess_dataset(path="./data/boli_nhs.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    updated_data = []
    for item in raw_data:
        disease = item.get("disease", "N/A")
        description = item.get("description", "N/A")
        symptoms_raw = item.get("symptoms", "N/A")

        # Dacă nu există simptome sau descriere, sărim peste acest element
        if not symptoms_raw or symptoms_raw == "N/A":
            continue

        # Extrage doar liniile care încep cu "-"
        lines = symptoms_raw.split("\n")
        bullet_symptoms = [line.strip() for line in lines if line.strip().startswith("-")]

        # Dacă nu sunt simptome de tip "- simptome", ignorăm elementul
        if not bullet_symptoms:
            continue

        # Alătură simptomele pe o singură linie
        cleaned_symptoms = " ".join(bullet_symptoms)

        

        updated_data.append({
            "disease": disease,
            "input": description,  
            "symptoms": cleaned_symptoms  
        })

    return pd.DataFrame(updated_data)

# Încărcarea și preprocesarea datasetului
dataset = load_and_preprocess_dataset(path="./data/boli_nhs.json")

# Încărcăm tokenizer-ul pre-antrenat
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

# Setăm pad_token la eos_token (End Of Sequence)
tokenizer.pad_token = tokenizer.eos_token

# Funcția pentru a preprocesa datele (tokenizare)
def preprocess_function(examples):
    return tokenizer(examples, padding=True, truncation=True)

# Aplicăm tokenizer-ul pe dataset (folosind simptomele și descrierea ca input)
tokenized_dataset = dataset["input"].apply(preprocess_function)

# Împărțim datasetul în seturi de antrenament și test
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Afișăm primele rânduri după preprocesare
print(train_data.head())

print("\n\n Cate boli avem",dataset.shape[0])  # Afișează numărul de rânduri
