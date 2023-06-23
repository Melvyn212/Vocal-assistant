from transformers import BertTokenizer, BertForSequenceClassification
import torch

class NLU:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)

    def predict_intent(self, text):
        inputs = self.tokenizer.encode_plus(text, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        outputs = self.model(input_ids, attention_mask=attention_mask)
        predicted_id = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = self.model.config.id2label[predicted_id]

        return predicted_label
nlu = NLU(model_name="bert-base-uncased")
