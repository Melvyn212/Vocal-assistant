import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

fs = 44100  # fréquence d'échantillonnage
duration = 5  # durée de l'enregistrement en secondes

print("Début de l'enregistrement...")
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
sd.wait()  # attendez que l'enregistrement se termine
print("Enregistrement terminé!")

# sauvegarder l'enregistrement dans un fichier .wav
write('output.wav', fs, myrecording)  
