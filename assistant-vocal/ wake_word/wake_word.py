import time
import pvporcupine
import pyaudio

class WakeWordDetector:
    def __init__(self, wake_words=['jarvis']):
        self.wake_words = wake_words
        self.porcupine = pvporcupine.create(keywords=wake_words)
        self.audio_stream = pyaudio.PyAudio().open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length)
        self.last_activity = time.time()

    def listen(self):
        while True:
            pcm = self.audio_stream.read(self.porcupine.frame_length)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            result = self.porcupine.process(pcm)
            if result >= 0:
                self.last_activity = time.time()
                return self.wake_words[result]

    def is_active(self):
        return time.time() - self.last_activity < 30
