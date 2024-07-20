import speech_recognition as sr
from pydub import AudioSegment
import os

def convert_to_wav(audio_path):
    """
    Convert audio file to WAV format if it is not already in WAV format.
    
    :param audio_path: Path to the audio file.
    :return: Path to the converted WAV file.
    """
    base, ext = os.path.splitext(audio_path)
    wav_path = f"{base}.wav"
    
    if ext.lower() != '.wav':
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format='wav')
    else:
        wav_path = audio_path
    
    return wav_path

def transcribe_audio(audio_path, language_code='en-US'):
    """
    Transcribe audio file using Google's speech recognition.
    
    :param audio_path: Path to the audio file.
    :param language_code: Language code for speech recognition.
    :return: Transcribed text.
    """
    recognizer = sr.Recognizer()
    text = ""

    try:
        wav_path = convert_to_wav(audio_path)  # Convert to WAV format if necessary
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language_code)
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        text = f"Could not request results from Google Speech Recognition service; {e}"
    except ValueError as e:
        text = f"Error processing the audio file; {e}"

    return text
