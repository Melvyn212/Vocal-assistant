from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import torchaudio

class ASR:
    def __init__(self):
        # Charger le tokenizer et le modèle pré-entraînés
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def transcribe(self, audio_file):
        # Charger le fichier audio
        waveform, sample_rate = torchaudio.load(audio_file)

        # Tokeniser l'audio
        input_values = self.tokenizer(waveform, return_tensors='pt').input_values

        # Passer l'audio par le modèle
        logits = self.model(input_values).logits

        # Prédire les tokens
        predicted_ids = torch.argmax(logits, dim=-1)

        # Décoder les tokens en texte
        transcription = self.tokenizer.decode(predicted_ids[0])

        return transcription
