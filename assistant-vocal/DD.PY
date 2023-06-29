from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialisez le tokenizer et le modèle
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Définissez la phrase que vous voulez reformuler
sentence = "Je suis content de te voir ce matin."
prompt = f"Comment pouvez-vous dire cela différemment: {sentence}?"

# Encodez la phrase pour le modèle GPT-2
inputs = tokenizer.encode(prompt, return_tensors='pt')

# Générez la sortie
outputs = model.generate(inputs, max_length=100, num_return_sequences=1, temperature=0.7)

# Décodez la sortie
decoded = tokenizer.decode(outputs[0])

print(decoded)
