from asr.googleAsr import voice_to_text
# from llm.llama2 import llama2_chat
from llm.gpt import qa_system
from MMSTTS.MMSTTS_class import MMSTTS
from asr.detect import detect_language

# text from ASR
text = voice_to_text()
lang_code="en" #detect_language(text)
#Translate text to english

#Fetch Text to LLM
print('Analyzing...')
context="You are an helpful AI assistant who has plenty of experience about indian law system. You are developed by a bunch of great minds at IRLP Lab from DAIICT"
llm_response = qa_system(text,context)
print('Response: ', llm_response)
#Text to Speech
print('Speaking...')
# text_to_voice(llm_response)
tts = MMSTTS(language=lang_code)
tts.synthesize(llm_response)
print(MMSTTS(llm_response,lang_code))