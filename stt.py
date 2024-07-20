import speech_recognition as sr
from pydub import AudioSegment
import io

def convert_to_wav(audio_file):
    """
    Convert audio file to WAV format if it is not already in WAV format.
    
    :param audio_file: File-like object of the audio file.
    :return: In-memory WAV file-like object.
    """
    audio = AudioSegment.from_file(audio_file)
    wav_io = io.BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)  # Reset stream position to the beginning
    return wav_io

def transcribe_audio(audio_file, language_code='en-US'):
    """
    Transcribe audio file using Google's speech recognition.
    
    :param audio_file: File-like object of the audio file.
    :param language_code: Language code for speech recognition.
    :return: Transcribed text.
    """
    recognizer = sr.Recognizer()
    text = ""
    
    try:
        wav_file = convert_to_wav(audio_file)  # Convert to WAV format if necessary
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language_code)
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        text = f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        text = f"Unexpected error; {e}"
    
    return text
