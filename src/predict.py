import sys
from model import get_model
import torch

tokenizer, model = get_model()
model.eval()

text = sys.argv[1]
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"Text: {text}")
print(f"Predicted label: {prediction}")
