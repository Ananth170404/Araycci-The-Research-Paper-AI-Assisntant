import speech_recognition as sr

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
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language_code)
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        text = f"Could not request results from Google Speech Recognition service; {e}"

    return text
