import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

class SoundRecorder:
    def __init__(self, fs=44100, duration=5):
        """
        Initialise une instance de SoundRecorder avec une fréquence d'échantillonnage et une durée données.

        Args:
            fs (int): Fréquence d'échantillonnage. Par défaut à 44100.
            duration (int): Durée de l'enregistrement en secondes. Par défaut à 5.
        """
        self.fs = fs
        self.duration = duration

    def record(self,sound_path='assistant-vocal\Soundsave\Savedsound\otput.wav'):
        """
        Commence l'enregistrement de l'audio et sauvegarde l'audio enregistré dans un fichier .wav.
        """
        print("Début de l'enregistrement...")
        myrecording = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=2)
        sd.wait()  # attendez que l'enregistrement se termine
        print("Enregistrement terminé!")
        # sauvegarder l'enregistrement dans un fichier .wav
        write(sound_path, self.fs, myrecording) 

        return sound_path
