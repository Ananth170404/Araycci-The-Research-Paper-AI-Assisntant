import pyaudio
import wave
import speech_recognition as sr
import threading
import IPython.display as ipd

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "output.wav"

audio = pyaudio.PyAudio()
frames = []
recording = False
stream = None
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
    global recording, stream, frames, record_thread
    if not recording:
        recording = True
        frames = []
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        print("Recording started. Speak into the microphone.")
        record_thread = threading.Thread(target=record_audio)
        record_thread.start()

def record_audio():
    global recording, stream, frames
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

def stop_recording():
    global recording, stream, record_thread
    if recording:
        recording = False
        record_thread.join()
        stream.stop_stream()
        stream.close()
        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        print("Recording stopped.")
        play_audio()
        text=transcribe_audio()
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
