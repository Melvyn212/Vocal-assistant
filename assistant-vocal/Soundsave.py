import speech_recognition as sr

class AudioRecorder:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def record_audio(self, filename, duration=5):
        with sr.Microphone() as source:
            print("Enregistrement en cours, parlez maintenant...")
            audio = self.recognizer.record(source, duration=duration)
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
            print("Enregistrement terminé et sauvegardé.")


# Création d'une instance de AudioRecorder
recorder = AudioRecorder()

# Nom du fichier où l'audio sera sauvegardé
filename = "mon_enregistrement.wav"

# Enregistrement de l'audio
# Le paramètre duration définit la durée de l'enregistrement en secondes
recorder.record_audio(filename, duration=10)
