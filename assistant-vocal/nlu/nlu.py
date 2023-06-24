from transformers import BertTokenizer, BertForSequenceClassification
import torch

class NLU:
    def __init__(self, model_name):
        """
        Initialise une instance de NLU (Natural Language Understanding).

        Args:
            model_name (str): Le nom du modèle pré-entraîné.
        """
        self.model_name = model_name
        # Initialiser le tokenizer et le modèle à partir du nom du modèle pré-entraîné.
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)

    def predict_intent(self, text):
        """
        Prédit l'intention de l'utilisateur à partir d'un texte.

        Args:
            text (str): Le texte à analyser.

        Returns:
            str: L'intention prédite.
        """
        # Encoder le texte à l'aide du tokenizer.
        inputs = self.tokenizer.encode_plus(text, return_tensors='pt')
        # Extraire les ids d'entrée et les masques d'attention.
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Faire une prédiction avec le modèle.
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # Extraire l'ID prédit en utilisant argmax pour sélectionner l'ID avec la plus grande logit.
        predicted_id = torch.argmax(outputs.logits, dim=1).item()
        # Obtenir l'étiquette correspondant à l'ID prédit.
        predicted_label = self.model.config.id2label[predicted_id]

        return predicted_label