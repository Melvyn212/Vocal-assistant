from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch

class ASR:
    def __init__(self):
        # Charger le tokenizer et le modèle pré-entraînés
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def transcribe(self, waveform):
        # Assurez-vous que la forme d'onde est un tensor PyTorch
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        # Tokeniser l'audio
        input_values = self.tokenizer(waveform, return_tensors='pt').input_values

        # Passer l'audio par le modèle
        logits = self.model(input_values).logits

        # Prédire les tokens
        predicted_ids = torch.argmax(logits, dim=-1)

        # Décoder les tokens en texte pour chaque prédiction dans le lot
        transcriptions = [self.tokenizer.decode(ids) for ids in predicted_ids]

        return transcriptions
