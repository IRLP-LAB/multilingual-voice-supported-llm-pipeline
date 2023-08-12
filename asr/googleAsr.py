import speech_recognition as sr

def voice_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)  # Use Google Speech Recognition engine
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print(f"Error: {str(e)}")

# Example usage
user_input = voice_to_text()