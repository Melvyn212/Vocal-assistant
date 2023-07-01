from transformers import GPT2LMHeadModel, GPT2Tokenizer
from asr import AudioTranscriber

class TextResponder:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate_response(self, prompt):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        return response

transcriber = AudioTranscriber()

# Chemin vers votre fichier audio
audio_file = "mon_enregistrement.wav"

# Utilisation de la méthode transcribe pour obtenir la transcription
transcription = transcriber.transcribe(audio_file)

# Affichage de la transcription
print(transcription)


# Création d'une instance de TextResponder
responder = TextResponder()

# Obtention d'une réponse à un texte d'invite
prompt = transcription
response = responder.generate_response(prompt)

# Affichage de la réponse
print(response)
