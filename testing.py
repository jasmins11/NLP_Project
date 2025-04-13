import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load model and tokenizer
model_path = "trained_model"
tokenizer_path = "trained_tokenizer"

# Încarcă modelul fine-tunat
tokenizer = GPT2Tokenizer.from_pretrained("trained_tokenizer")
model = GPT2LMHeadModel.from_pretrained("trained_model")

# Setezi device-ul
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

symptome_input = "- shortness of breath - chest tightness - coughing"

prompt = f"Symptoms: {symptome_input}\nThis condition is likely:"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

output_sequences = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print("Modelul sugerează:\n")
print(generated_text)

