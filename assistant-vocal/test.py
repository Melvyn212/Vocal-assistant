
from Soundsave.Soundsave import SoundRecorder


recorder = SoundRecorder(fs=44100, duration=10)  # Crée une instance de SoundRecorder avec une fréquence d'échantillonnage de 44100 Hz et une durée de 10 secondes

output_path = recorder.record()  # Lance l'enregistrement et récupère le chemin du fichier enregistré

print("Fichier enregistré :", output_path)