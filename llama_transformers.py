from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers

print(transformers.__file__)  # Should point to YOUR editable install
# Load tokenizer and model
model_name_or_path = "Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_safetensors=True)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



prompt = "I'm from Paris and my favorite neighbourhood is "
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

output = ""
for i in range(1):
    with torch.no_grad():
        output_logits = model(input_ids)
        logits = output_logits.logits  # Access logits properly
        last_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(last_token_logits, dim=-1)

        # Append new token
        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(1)), dim=1)

        decoded_token = tokenizer.decode(next_token_id, skip_special_tokens=True)
        output += decoded_token

print("SENTENCE ==> ", output)
