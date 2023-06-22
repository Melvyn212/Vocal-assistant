# Assistant Vocal Personnel

Ce projet implémente un assistant vocal personnel à l'aide de PyTorch et de la suite de modèles HuggingFace. L'assistant peut reconnaître la parole, comprendre le langage naturel, gérer les dialogues, générer du langage naturel et synthétiser la parole.

## Structure du projet

Le projet est organisé en plusieurs dossiers, chacun se concentrant sur une partie spécifique du pipeline de l'assistant vocal :

- `asr/` - Reconnaissance Automatique de la Parole (ASR)
- `nlu/` - Compréhension du Langage Naturel (NLU)
- `dialogue_management/` - Gestion du Dialogue
- `nlg/` - Génération du Langage Naturel (NLG)
- `tts/` - Synthèse de Texte en Parole (TTS)
- `wake_word/` - Détection du Mot d'Activation

Chaque dossier contient un script Python pour sa fonction spécifique, ainsi qu'un dossier `models/` pour stocker les modèles pré-entraînés pertinents.

## Fonctionnement

L'assistant vocal fonctionne en suivant un pipeline d'étapes :

1. **Écoute:** L'assistant écoute les entrées audio de l'utilisateur. Si l'assistant détecte un mot d'activation (par exemple, "hey assistant"), il passe à l'étape suivante.

2. **Reconnaissance Automatique de la Parole (ASR):** L'assistant convertit l'entrée audio en texte à l'aide du modèle ASR.

3. **Compréhension du Langage Naturel (NLU):** L'assistant utilise le modèle NLU pour comprendre le sens du texte.

4. **Gestion du Dialogue:** En utilisant le contexte de la conversation actuelle, l'assistant utilise le modèle de gestion du dialogue pour déterminer la meilleure façon de répondre à la demande de l'utilisateur.

5. **Génération du Langage Naturel (NLG):** L'assistant utilise le modèle NLG pour générer une réponse textuelle à la demande de l'utilisateur.

6. **Synthèse de la Parole (TTS):** L'assistant utilise le modèle TTS pour convertir la réponse textuelle en parole. 

7. Le processus retourne alors à l'étape 1, en écoutant à nouveau les entrées de l'utilisateur. Si à tout moment l'utilisateur dit "Arrête d'écouter" ou un autre mot d'arrêt défini, l'assistant arrête d'écouter et le cycle se termine.

## Utilisation

Pour utiliser l'assistant vocal, exécutez le script principal comme suit :

```python
if __name__ == '__main__':
    config = {
        'asr_model_path': 'asr/models/',
        'nlu_model_path': 'nlu/models/',
        'dialogue_model_path': 'dialogue_management/models/',
        'nlg_model_path': 'nlg/models/',
        'tts_model_path': 'tts/models/',
        'wake_words': ['hey assistant', 'ok assistant']
    }
    assistant = VoiceAssistant(config)
    assistant.run()
```

Dans ce script, `config` est un dictionnaire qui contient les chemins vers les modèles pré-entraînés et les mots d'activation