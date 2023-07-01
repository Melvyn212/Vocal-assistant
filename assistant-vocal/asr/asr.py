import speech_recognition as sr

class AudioTranscriber:
    def __init__(self, language="fr-FR"):
        self.recognizer = sr.Recognizer()
        self.language = language

    def transcribe(self, audio_file):
        with sr.AudioFile(audio_file) as source:
            audio = self.recognizer.record(source)
            try:
                transcription = self.recognizer.recognize_google(audio, language=self.language)
                return transcription
            except sr.UnknownValueError:
                return "Google Speech Recognition n'a pas compris l'audio"
            except sr.RequestError as e:
                return f"Le service de reconnaissance vocale Google a échoué; {e}"


# Création d'une instance de AudioTranscriber
transcriber = AudioTranscriber()

# Chemin vers votre fichier audio
audio_file = "mon_enregistrement.wav"

# Utilisation de la méthode transcribe pour obtenir la transcription
transcription = transcriber.transcribe(audio_file)

# Affichage de la transcription
print(transcription)
