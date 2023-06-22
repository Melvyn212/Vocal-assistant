from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class DialogueManager:
    def __init__(self, model_name, emotion_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.emotion_predictor = pipeline('text-classification', model=emotion_model_name)
        self.sessions = {}

    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "emotion_state": "neutral",
                "error_count": 0
            }
        return self.sessions[session_id]

    def update_emotion(self, session_id):
        session = self.get_session(session_id)
        # Predict the emotion based on the conversation history
        dialogue_history = [turn for history in session['history'] for turn in history if turn]
        input_text = self.tokenizer.eos_token.join(dialogue_history)
        emotion = self.emotion_predictor(input_text)
        session["emotion_state"] = emotion[0]['label']

    def get_response(self, session_id, user_input):
        session = self.get_session(session_id)

        # If too many errors occurred, reset the state
        if session['error_count'] > 5:
            session['history'] = []
            session['emotion_state'] = "neutral"
            session['error_count'] = 0
            return "I'm sorry, I seem to be having trouble understanding. Let's start over."

        # Update the emotion state based on the user's input
        self.update_emotion(session_id)

        # Add the emotion state to the user's input
        user_input_with_emotion = f"{session['emotion_state']}: {user_input}"

        # Concatenate the conversation history and the user's current input
        dialogue_history = [turn for history in session['history'] for turn in history if turn]
        dialogue_history.append(user_input_with_emotion)
        input_text = self.tokenizer.eos_token.join(dialogue_history)

        try:
            # Generate a response from the model
            inputs = self.tokenizer.encode(input_text, return_tensors='pt')
            outputs = self.model.generate(inputs, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(outputs[0])

            # Update the conversation history with the model's response
            session['history'].append((user_input_with_emotion, response))
        except Exception as e:
            session['error_count'] += 1
            return f"I'm sorry, I encountered an error: {str(e)}"

        return response
