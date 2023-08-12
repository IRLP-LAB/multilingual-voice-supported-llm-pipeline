import langid

def detect_language(text):
    detected_language, confidence = langid.classify(text)
    return detected_language

if __name__ == "__main__":
    input_text = " ଓଡ଼ିଆ ଭାଷା ଭାରତରେ ମୁଖ୍ଯତମା ଓରିୟାଭାଷାର ନାମେ ପ୍ରସିଦ୍ଧି ରଖୁଛି। ଏହା ଓଡ଼ିଶା ର ସର୍ବାଧିକ ବୋଲିଆଉଠା ଭାଷା ହୋଇଛି। ଓଡ଼ିଆ ଭାଷାର ସମାଚାର, ବିଜ୍ଞାନ, ସାହିତ୍ୟ, ଧର୍ମବିଚାର, ଭୂଗୋଳ, ଇତିହାସ, ପ୍ରବନ୍ଧ, ପଦ୍ଯ, ଗୀତ, ନାଟକ, ଚଳଚ୍ଚିତ୍ର ଇତ୍ୟାଦି ସକଳ ଶ୍ରେଷ୍ଠ ଭାଷାରେ ଗୁଣାତିଶୟ ଧାରାବାରକର ବାଟରେ ରଖିଯାଉଥିବାରେ ବିଶେଷ ଅଭିନନ୍ଦନ ରଖୁଛି।"
    language = detect_language(input_text)
    print(f"Detected language: {language}")
