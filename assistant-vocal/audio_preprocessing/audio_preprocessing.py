import numpy as np
import pandas as pd
import torchaudio

class AudioPreprocessor:
    def __init__(self, sample_rate, threshold):
        """
        Constructeur de la classe AudioPreprocessor.
        
        Args:
            sample_rate (int): le taux d'échantillonnage de l'audio à prétraiter.
            threshold (float): le seuil utilisé pour détecter les parties utiles de l'audio.
        """
        self.sample_rate = sample_rate
        self.threshold = threshold

    def envelope(self, signal):
        """
        Crée un masque pour le signal audio en fonction du seuil.

        Args:
            signal (np.array): le signal audio à masquer.

        Returns:
            np.array: un masque indiquant les parties du signal à conserver.
        """
        mask = []
        signal = pd.Series(signal).apply(np.abs) 
        signal_mean = signal.rolling(window = int(self.sample_rate/10), min_periods = 1, center = True).mean()
        for mean in signal_mean:
            if mean > self.threshold:
                mask.append(True)
            else:
                mask.append(False)
        return np.array(mask)

    def load_audio(self, audio_file):
        """
        Charge un fichier audio et le convertit en une forme d'onde.

        Args:
            audio_file (str): le chemin vers le fichier audio à charger.

        Returns:
            np.array: la forme d'onde du fichier audio.
        """
        waveform, _ = torchaudio.load(audio_file)
        return waveform.numpy()

    def preprocess_audio(self, audio_file):
        """
        Charge un fichier audio, le convertit en une forme d'onde, et applique le masque créé par la fonction enveloppe.

        Args:
            audio_file (str): le chemin vers le fichier audio à prétraiter.

        Returns:
            np.array: le signal audio prétraité.
        """
        signal = self.load_audio(audio_file)
        mask = self.envelope(signal)
        return signal[mask]
