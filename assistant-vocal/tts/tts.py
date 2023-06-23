import torch
from torch import nn
from tacotron2.model import Tacotron2
from waveglow.glow import WaveGlow
import denoiser
from tacotron2.text import text_to_sequence
import numpy as np

class VoiceSynthesizer:
    def __init__(self, tacotron_model_path, waveglow_model_path):
        """
        Initialise une instance de VoiceSynthesizer avec les modèles Tacotron2 et WaveGlow.

        Args:
            tacotron_model_path (str): Le chemin vers le modèle pré-entraîné Tacotron2.
            waveglow_model_path (str): Le chemin vers le modèle pré-entraîné WaveGlow.
        """
        # Charger le modèle Tacotron2
        tacotron_checkpoint = torch.load(tacotron_model_path, map_location=torch.device('cpu'))
        self.tacotron2 = Tacotron2(tacotron_checkpoint['config'])
        self.tacotron2.load_state_dict(tacotron_checkpoint['model'])
        self.tacotron2.eval().half()

        # Charger le modèle WaveGlow
        waveglow_checkpoint = torch.load(waveglow_model_path, map_location=torch.device('cpu'))
        self.waveglow = WaveGlow(waveglow_checkpoint['config'])
        self.waveglow.load_state_dict(waveglow_checkpoint['model'])
        self.waveglow.eval().half()

        # Initialiser le débruiteur
        self.denoiser = denoiser(self.waveglow)

    def synthesize(self, text, sigma=0.666):
        """
        Synthétise un texte en voix parlée.

        Args:
            text (str): Le texte à synthétiser.
            sigma (float): La force du bruit d'entrée pour le modèle WaveGlow.

        Returns:
            np.array: Le signal audio de la voix synthétisée.
        """
        # Convertir le texte en une séquence d'IDs de phonèmes
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

        # Obtenir les sorties du modèle Tacotron2
        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron2.inference(sequence)
        
        # Synthétiser la voix avec le modèle WaveGlow
        with torch.no_grad():
            audio = self.waveglow.inference(mel_outputs_postnet, sigma=sigma)
            # Utiliser le débruiteur pour enlever le bruit du signal audio
            audio_denoised = self.denoiser(audio, strength=0.01)[:, 0]

        return audio_denoised.numpy()
