from lingua import Language, LanguageDetectorBuilder

languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

def detect_long_text(text, chunk_size=100):
    try:
        lang=detector.detect_language_of(text)
        print(lang)
        print(lang.iso_code_639_1)
        print(lang.iso_code_639_3.name.lower())


        return lang
    
    except Exception as e:
        return ""

detect_long_text("Bonjour")