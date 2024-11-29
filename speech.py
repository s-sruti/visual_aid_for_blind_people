import speech_recognition as sr
import pyttsx3
import logging
import numpy as np
import sounddevice as sd
from transformers import pipeline


import whisper

class speech_to_text():
    def __init__(self):
       en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
    
    #    self.model = pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
       self.model=whisper.load_model("tiny")
       self.engine = pyttsx3.init()
       self.engine.setProperty('voice', en_voice_id)
    
    def record_audio(self, duration=5):
        fs = 16000  
        print("Recording...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  
        print("Recording complete.")
        return audio.flatten()  

    def recognize_speech_from_mic(self):
        """Recognizes speech from the microphone using Whisper."""
        audio = self.record_audio(duration=5) 
        result = self.model.transcribe(audio,language="en")
        return result['text']
       
    def text_speech(self, cleaned_text):
        self.engine.say(cleaned_text);
        self.engine.runAndWait();

