import sounddevice as sd
import wave
import speech_recognition as sr
import threading
import IPython.display as ipd
import numpy as np
from scipy.io.wavfile import write

# Parameters
FORMAT = np.int16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "output.wav"

frames = []
recording = False
record_thread = None
language = 'en-US'

# Language map
language_map = {
    'English': 'en-US',
    'German': 'de-DE',
    'Spanish': 'es-ES',
    'French': 'fr-FR'
}

def start_recording():
    global recording, frames, record_thread
    if not recording:
        recording = True
        frames = []
        print("Recording started. Speak into the microphone.")
        record_thread = threading.Thread(target=record_audio)
        record_thread.start()

def record_audio():
    global recording, frames
    while recording:
        data = sd.rec(CHUNK, samplerate=RATE, channels=CHANNELS, dtype=FORMAT)
        sd.wait()
        frames.append(data)

def stop_recording():
    global recording, record_thread
    if recording:
        recording = False
        record_thread.join()
        frames_np = np.concatenate(frames, axis=0)
        write(WAVE_OUTPUT_FILENAME, RATE, frames_np)
        print("Recording stopped.")
        play_audio()
        text = transcribe_audio()
        return text

def play_audio():
    ipd.display(ipd.Audio(WAVE_OUTPUT_FILENAME))

def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
        audio_data = recognizer.record(source)
        try:
            # Recognize speech using Google's speech recognition
            text = recognizer.recognize_google(audio_data, language=language)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return text
