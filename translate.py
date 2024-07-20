import os
from together import Together
from gtts import gTTS
import io
from tokens import token_size

def translate(text, lang):

  #SETTING ENVIRONMENT VARS
  os.environ['TOGETHER_API_KEY'] = "dc4921bdc25d60750f8610d2f7212a8c26b6b8949450d31387fba18ee42a0b07"


  #SETTING UP MODEL
  client = Together(api_key=os.environ.get(os.environ['TOGETHER_API_KEY']))

  #SETTING SYSTEM PROMPT

  prompt="You are a translator bot, translate "+text+" to "+lang+" Only give the translated text."

  messages=[
            {
            "role": "system",
            "content": "You are a translator bot. ONLY GIVE THE TRANSLATED OUTPUT AND NOTHING ELSE"
          },
          {
            "role": "user",
            "content": prompt
          }
  ]

  max_t=8192-token_size(prompt)

  #GET RESPONSE
  response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-chat-hf",
    messages=messages,
    max_tokens=max_t,
    temperature=0.2,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1.3,
    stop=["<|eot_id|>"],
    stream=True
  )

  response_text = ""
  for chunk in response:
    for choice in chunk.choices:
      if choice.text:
        response_text += choice.text

  return response_text

def generate_audio(text, lang):
    languages={"English":"en", "French":"fr", "Spanish": "es"}
    tts = gTTS(text=text, lang=languages[lang])
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return audio_io
# lang=input("Language: ")
# text="Should we Stop Training More Monolingual Models, and Simply Use Machine Translation Instead? (2021-04-21T10:21:24Z)\nAuthors: Tim Isbister, Fredrik Carlsson, Magnus Sahlgren\nSummary: Most work in NLP makes the assumption that it is desirable to develop solutions in the native language in question. There is consequently a strong trend towards building native language models even for low-resource languages. This paper questions this development, and explores the idea of simply translating the data into English, thereby enabling the use of pretrained, and large-scale, English language models. We demonstrate empirically that a large English language model coupled with modern machine translation outperforms native language models in most Scandinavian languages. The exception to this is Finnish, which we assume is due to inferior translation quality. Our results suggest that machine translation is a mature technology, which raises a serious counter-argument for training native language models for low-resource languages. This paper therefore strives to make a provocative but important point. As English language models are improving at an unprecedented pace, which in turn improves machine translation, it is from an empirical and environmental stand-point more effective to translate data from low-resource languages into English, than to build language models for such languages."

# print(translate(text, lang))