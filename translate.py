import os
from together import Together
from gtts import gTTS
import io
from tokens import token_size

def translate(text, lang):
    # Load the Together API key from the environment variables
    together_api_key = os.getenv("TOGETHER_API_KEY")

    # SETTING UP MODEL
    client = Together(api_key=together_api_key)

    # SETTING SYSTEM PROMPT
    prompt = "You are a translator bot, translate " + text + " to " + lang + " Only give the translated text."

    messages = [
        {
            "role": "system",
            "content": "You are a translator bot. ONLY GIVE THE TRANSLATED OUTPUT AND NOTHING ELSE"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    max_t = 8192 - token_size(prompt)

    # GET RESPONSE
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=messages,
        max_tokens=max_t,
        temperature=0.2,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.3,
        stop=[""],
        stream=True
    )

    response_text = ""
    for chunk in response:
        for choice in chunk.choices:
            if choice.text:
                response_text += choice.text

    return response_text

def generate_audio(text, lang):
    languages = {"English": "en", "French": "fr", "Spanish": "es"}
    tts = gTTS(text=text, lang=languages[lang])
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return audio_io
