

from asr import ASR

# Créer une instance de la classe ASR
asr = ASR()

# Transcrire un fichier audio
transcription = asr.transcribe('path_to_your_audio_file.wav')

print('Transcription :', transcription)
