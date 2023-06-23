import numpy as np
import pandas as pd
import torchaudio

class AudioPreprocessor:
    def __init__(self, sample_rate, threshold):
        self.sample_rate = sample_rate
        self.threshold = threshold

    def envelope(self, signal):
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
        waveform, _ = torchaudio.load(audio_file)
        return waveform.numpy()

    def preprocess_audio(self, audio_file):
        signal = self.load_audio(audio_file)
        mask = self.envelope(signal)
        return signal[mask]
