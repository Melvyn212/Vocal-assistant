
from Soundsave.Soundsave import SoundRecorder
import soundfile as sf
from asr.asr import ASR
import librosa
import numpy as np
from nlu.nlu import NLU



recorder = SoundRecorder(fs=44100, duration=5)  # Crée une instance de SoundRecorder avec une fréquence d'échantillonnage de 44100 Hz et une durée de 10 secondes

output_path = recorder.record()  # Lance l'enregistrement et récupère le chemin du fichier enregistré

print("Fichier enregistré :", output_path)


# Charger l'audio
waveform, sr = librosa.load(output_path, sr=16000)

# Normaliser à des valeurs entre -1 et 1
waveform = waveform / np.abs(waveform).max()

asr = ASR()  # Crée une instance de la classe ASR

# Passer à la méthode transcribe
transcriptions = asr.transcribe(waveform)


# Afficher les transcriptions
for transcription in transcriptions:
    print(transcription)

# Initialize the NLU
# Créer une instance de NLU avec le modèle BERT de base non censuré.
nlu = NLU(model_name="bert-base-uncased")
predicted_intent = nlu.predict_intent(transcription)
print(f"The predicted intent of the sentence is: {predicted_intent}")
