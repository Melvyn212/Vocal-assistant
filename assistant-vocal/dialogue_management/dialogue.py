from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class DialogueManager:
    def __init__(self, model_name, emotion_model_name):
        """
        Initialise une instance de DialogueManager.

        Args:
            model_name (str): Le nom du modèle de dialogue pré-entraîné.
            emotion_model_name (str): Le nom du modèle d'émotion pré-entraîné.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.emotion_predictor = pipeline('text-classification', model=emotion_model_name)
        self.sessions = {}

    def get_session(self, session_id):
        """
        Récupère une session de dialogue.

        Args:
            session_id (str): L'ID de la session.

        Returns:
            dict: La session de dialogue.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "emotion_state": "neutral",
                "error_count": 0
            }
        return self.sessions[session_id]

    def update_emotion(self, session_id):
        """
        Met à jour l'état émotionnel de la session en fonction de l'historique de la conversation.

        Args:
            session_id (str): L'ID de la session.
        """
        session = self.get_session(session_id)
        dialogue_history = [turn for history in session['history'] for turn in history if turn]
        input_text = self.tokenizer.eos_token.join(dialogue_history)
        emotion = self.emotion_predictor(input_text)
        session["emotion_state"] = emotion[0]['label']

    def get_response(self, session_id, user_input):
        """
        Génère une réponse du modèle de dialogue en fonction de l'entrée de l'utilisateur.

        Args:
            session_id (str): L'ID de la session.
            user_input (str): L'entrée de l'utilisateur.

        Returns:
            str: La réponse du modèle de dialogue.
        """
        session = self.get_session(session_id)

        if session['error_count'] > 5:
            session['history'] = []
            session['emotion_state'] = "neutral"
            session['error_count'] = 0
            return "I'm sorry, I seem to be having trouble understanding. Let's start over."

        self.update_emotion(session_id)

        user_input_with_emotion = f"{session['emotion_state']}: {user_input}"

        dialogue_history = [turn for history in session['history'] for turn in history if turn]
        dialogue_history.append(user_input_with_emotion)
        input_text = self.tokenizer.eos_token.join(dialogue_history)

        try:
            inputs = self.tokenizer.encode(input_text, return_tensors='pt')
            outputs = self.model.generate(inputs, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(outputs[0])

            session['history'].append((user_input_with_emotion, response))
        except Exception as e:
            session['error_count'] += 1
            return f"I'm sorry, I encountered an error: {str(e)}"

        return response
