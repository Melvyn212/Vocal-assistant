import struct
import time
import pvporcupine
import pyaudio

class WakeWordDetector:
    def __init__(self, wake_words=['jarvis']):
        """
        Initialise une instance de WakeWordDetector avec des mots de réveil spécifiques.

        Args:
            wake_words (list, optional): Une liste de mots de réveil. Par défaut à ['jarvis'].
        """
        # Créer une instance Porcupine avec les mots de réveil
        self.wake_words = wake_words
        self.porcupine = pvporcupine.create(keywords=wake_words)

        # Ouvrir un flux audio avec PyAudio pour écouter les mots de réveil
        self.audio_stream = pyaudio.PyAudio().open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length)
        
        # Garder une trace du dernier moment où un mot de réveil a été détecté
        self.last_activity = time.time()

    def listen(self):
        """
        Écoute en continu les mots de réveil et renvoie le mot de réveil détecté.
        
        Returns:
            str: Le mot de réveil détecté.
        """
        while True:
            # Lire les données audio du flux
            pcm = self.audio_stream.read(self.porcupine.frame_length)
            
            # Décoder les données audio
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            
            # Utiliser Porcupine pour traiter les données audio
            result = self.porcupine.process(pcm)

            # Si un mot de réveil a été détecté, mettre à jour l'heure de la dernière activité
            # et renvoyer le mot de réveil détecté
            if result >= 0:
                self.last_activity = time.time()
                return self.wake_words[result]

    def is_active(self):
        """
        Vérifie si un mot de réveil a été détecté dans les 30 dernières secondes.

        Returns:
            bool: True si un mot de réveil a été détecté dans les 30 dernières secondes, sinon False.
        """
        return time.time() - self.last_activity < 30
