from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLU:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict_intent(self, text):
        inputs = self.tokenizer.encode(text, return_tensors='pt')
        outputs = self.model(inputs)
        predicted_id = outputs.logits.argmax(-1).item()
        return self.model.config.id2label[predicted_id]
