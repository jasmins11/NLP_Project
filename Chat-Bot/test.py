from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


model_path = "trained_model"
tokenizer_path = "trained_tokenizer"

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
model.eval()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model.to(device)


while True:
    input_text = input("\n Disease: ")
    if input_text.lower() == "exit":
        break

    
    prompt = f"Disease: {input_text} | Symptoms:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)


    output = model.generate(
        input_ids,
        max_length=512,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n Complete:")
    print(generated_text)
