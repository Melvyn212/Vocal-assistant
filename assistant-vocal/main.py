from asr.asr import ASR
from nlu.nlu import NLU
from dialogue_management.dialogue import DialogueManager
from nlg.nlg import NLG
from tts.tts import TTS
from wakeword.wake_word import WakeWordDetector
from audio_preprocessing.audio_preprocessing import AudioPreprocessor
import pyaudio

class VoiceAssistant:
    def __init__(self, config):
        self.asr = ASR(config['asr_model_path'])
        self.nlu = NLU(config['nlu_model_path'])
        self.dialogue_manager = DialogueManager(config['dialogue_model_path'])
        self.nlg = NLG(config['nlg_model_path'])
        self.tts = TTS(config['tts_model_path'])
        self.wake_word_detector = WakeWordDetector(config['wake_words'])
        self.audio_preprocessor = AudioPreprocessor(config['audio_preprocessing_model_path'])
        self.audio = pyaudio.PyAudio()
        self.listening = True

    def run(self):
        self.tts.speak("Hi, what can I do for you?")
        while self.listening:
            wake_word = self.wake_word_detector.listen()
            if wake_word and self.wake_word_detector.is_active():
                raw_audio = self.listen_for_audio()
                preprocessed_audio = self.audio_preprocessor.process(raw_audio)
                text = self.asr.transcribe(preprocessed_audio)
                if "stop listening" in text:
                    self.listening = False
                    print('Listening stopped')
                    continue
                intent, entities = self.nlu.process(text)
                response = self.dialogue_manager.respond(intent, entities)
                generated_text = self.nlg.generate(response)
                self.tts.speak(generated_text)

    def listen_for_audio(self):
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []
        silence_count = 0
        while True:
            data = stream.read(1024)
            frames.append(data)
            if self.is_silent(data):
                silence_count += 1
                if silence_count > 32:
                    break
            else:
                silence_count = 0
        stream.stop_stream()
        stream.close()
        return b''.join(frames)

    def is_silent(self, data):
        return max(data) < 100

if __name__ == '__main__':
    config = {
        'asr_model_path': 'asr/models/',
        'nlu_model_path': 'nlu/models/',
        'dialogue_model_path': 'dialogue_management/models/',
        'nlg_model_path': 'nlg/models/',
        'tts_model_path': 'tts/models/',
        'wake_words': ['hey assistant', 'ok assistant'],
        'audio_preprocessing_model_path': 'audio_preprocessing/models/',
    }
    assistant = VoiceAssistant(config)
    assistant.run()
