from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_model(model_name="distilbert-base-uncased", num_labels=6):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


