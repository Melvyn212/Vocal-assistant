import torch
from torch import nn
from tacotron2.model import Tacotron2
from waveglow.glow import WaveGlow
from denoiser import Denoiser
from tacotron2.text import text_to_sequence

class VoiceSynthesizer:
    def __init__(self, tacotron_model_path, waveglow_model_path):
        tacotron_checkpoint = torch.load(tacotron_model_path, map_location=torch.device('cpu'))
        self.tacotron2 = Tacotron2(tacotron_checkpoint['config'])
        self.tacotron2.load_state_dict(tacotron_checkpoint['model'])
        self.tacotron2.eval().half()

        waveglow_checkpoint = torch.load(waveglow_model_path, map_location=torch.device('cpu'))
        self.waveglow = WaveGlow(waveglow_checkpoint['config'])
        self.waveglow.load_state_dict(waveglow_checkpoint['model'])
        self.waveglow.eval().half()

        self.denoiser = Denoiser(self.waveglow)

    def synthesize(self, text, sigma=0.666):
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron2.inference(sequence)
        with torch.no_grad():
            audio = self.waveglow.inference(mel_outputs_postnet, sigma=sigma)
            audio_denoised = self.denoiser(audio, strength=0.01)[:, 0]

        return audio_denoised.numpy()
