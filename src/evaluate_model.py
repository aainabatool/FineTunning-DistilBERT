from data_loader import get_dataset
from model import get_model
from transformers import Trainer
import evaluate

dataset = get_dataset()
tokenizer, model = get_model()

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

accuracy = evaluate.load("accuracy")

trainer = Trainer(model=model)
metrics = trainer.evaluate(encoded_dataset["test"])
print(metrics)
