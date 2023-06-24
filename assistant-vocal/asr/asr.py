import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

class ASR:
    def __init__(self):
        """
        Constructeur de la classe ASR. Charge le tokenizer et le modèle pré-entraînés.
        """
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def transcribe(self, waveform):
        """
        Transcrit une forme d'onde audio en texte.

        Args:
            waveform (np.array ou torch.Tensor): La forme d'onde audio à transcrire.

        Returns:
            list[str]: La transcription du fichier audio.
        """
        # Assurez-vous que la forme d'onde est un tensor PyTorch
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        # Process the waveform
        inputs = self.feature_extractor(waveform, return_tensors='pt', sampling_rate=16000)
        input_values = inputs.input_values

        # Passer l'audio par le modèle
        logits = self.model(input_values).logits

        # Prédire les tokens
        predicted_ids = torch.argmax(logits, dim=-1)

        # Décoder les tokens en texte pour chaque prédiction dans le lot
        transcriptions = [self.tokenizer.decode(ids) for ids in predicted_ids]

        return transcriptions
