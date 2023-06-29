
from Soundsave.Soundsave import SoundRecorder
import soundfile as sf
from asr.asr import ASR
import librosa
import numpy as np
from nlu.nlu import NLU



recorder = SoundRecorder(fs=44100, duration=5)  # Crée une instance de SoundRecorder avec une fréquence d'échantillonnage de 44100 Hz et une durée de 10 secondes

output_path = recorder.record()  # Lance l'enregistrement et récupère le chemin du fichier enregistré

print("Fichier enregistré :", output_path)


import speech_recognition as sr

# Créer un objet Recognizer
r = sr.Recognizer()

# Ouvrir le fichier audio
with sr.AudioFile(output_path) as source:
    # Lire l'audio à partir du fichier
    audio_data = r.record(source)
    # Transcrire l'audio en texte
    text = r.recognize_google(audio_data)
    print(text)

