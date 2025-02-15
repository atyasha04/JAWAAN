import subprocess
from deep_translator import GoogleTranslator

def process_audio(file_path, model='medium'):
    # Check if the input file exists
    if not file_path.endswith('.mp3'):
        print("Error: The file must be an MP3 file.")
        return None, None, None

    # Transcribe the audio file using Whisper
    result = subprocess.run(['whisper', file_path, '--model', model], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    lines = result.stdout.splitlines()

    language = ""
    transcription = ""

    for line in lines:
        if line.startswith("Detected language:"):
            language = line.split(":")[1].strip()
        elif "-->" in line:
            transcription += line.split("]")[1].strip() + " "

    # Translate the transcription to English using Google Translator
    def translate_to_english(text):
        try:
            translator = GoogleTranslator(source='auto', target='en')
            return translator.translate(text)
        except Exception as e:
            return f"Error: {str(e)}"

    translated_text = translate_to_english(transcription)

    # Print results
    print(f"Detected Language: {language}")
    print(f"Transcription: {transcription}")
    print(f"Translated Text: {translated_text}")

    return language, transcription, translated_text

# Example usage:
file_path = '/content/output_audio.mp3'  # Provide your MP3 file path
process_audio(file_path)