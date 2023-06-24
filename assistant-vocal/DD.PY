from transformers import T5ForConditionalGeneration, T5Tokenizer

MAX_LENGTH=512

# Initialisez le tokenizer et le modèle
tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=MAX_LENGTH)
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Définissez la phrase que vous voulez reformuler
sentence = "MORNING  HAPPY TO SEE YOM"
sentence = "paraphrase: " + sentence

# Encodez la phrase pour le modèle T5
inputs = tokenizer.encode(sentence, return_tensors="pt")

# Générez la sortie
outputs = model.generate(inputs, max_length=40, num_return_sequences=1, temperature=0.7)

# Décodez la sortie
decoded = tokenizer.decode(outputs[0])

print(decoded)
