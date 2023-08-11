import os
from MMSTTS_class import MMSTTS

def main():
    
    tts = MMSTTS(language="eng")  # replace 'native_language_code' with the actual language code

    translated_text = """We at IRLP Lab, DA-IICT are working in the areas of Information Retrieval and Natural Language Processing focused on Indic Languages. We are currently working in the following areas: 
    1. Machine Translation for Indic languages
    2. Summarization and evaluation of news articles
    3. News recommendation system
    4. Stock market movement prediction
    5. Hate speech detection
    6. Biomedical Information Retrieval
    7. Exploring applications of AI in legal domain

    We are also home to various research tasks in information extraction, statistical machine learning, Deep Learning, Natural Language Modeling and Natural Language Understanding.

    Courses
    We offer Information Retrieval (IT550) and Natural Language Processing (IT412) courses in autumn and winter semesters respectively at DA-IICT."""

    # Generate the audio numpy array
    audio_array = tts.synthesize(translated_text)

    # Save the synthesized audio to 'output.wav'
    tts.save_to_file(audio_array, 'output_new.wav')

if __name__ == "__main__":
    main()