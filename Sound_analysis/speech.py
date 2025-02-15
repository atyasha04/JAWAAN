import subprocess
from deep_translator import GoogleTranslator
import os
import tensorflow as tf
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow_hub as hub
import io
import base64
from pydub import AudioSegment

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# Load YAMNet model from TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Load class names for YAMNet
class_names_df = pd.read_csv('yamnet_class_map.csv')
class_names = class_names_df['display_name'].values

# Functions
def extract_audio(video_path, audio_path):
    """Extract audio from a video file."""
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path]
    subprocess.run(command, check=True)

def convert_to_wav(file_path):
    """Convert audio file to WAV format."""
    audio = AudioSegment.from_file(file_path)
    output_file_path = os.path.splitext(file_path)[0] + ".wav"
    audio.export(output_file_path, format="wav")
    return output_file_path

def load_wav_16k_mono(filename):
    """Load an audio file and resample it to 16 kHz mono."""
    waveform, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return waveform, sample_rate

def plot_audio_features(waveform, sample_rate):
    """Plot various audio features and return as base64-encoded images."""
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Audio Feature Visualization', fontsize=16)
    cmap = 'plasma'

    # Plot Sound Wave
    axs[0, 0].plot(np.arange(len(waveform)) / sample_rate, waveform, color='blue')
    axs[0, 0].set_title('Sound Wave')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)

    # Plot Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
    img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log', ax=axs[0, 1], cmap=cmap)
    axs[0, 1].set_title('Spectrogram')
    fig.colorbar(img, ax=axs[0, 1], format='%+2.0f dB')

    # Plot Mel Spectrogram
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=sample_rate), ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sample_rate, x_axis='time', y_axis='mel', ax=axs[1, 0], cmap=cmap)
    axs[1, 0].set_title('Mel Spectrogram')
    fig.colorbar(img, ax=axs[1, 0], format='%+2.0f dB')

    # Plot Chroma Features
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
    img = librosa.display.specshow(chroma, sr=sample_rate, x_axis='time', y_axis='chroma', ax=axs[1, 1], cmap=cmap)
    axs[1, 1].set_title('Chroma Features')
    fig.colorbar(img, ax=axs[1, 1])

    # Plot Constant-Q Transform (CQT)
    cqt_db = librosa.amplitude_to_db(np.abs(librosa.cqt(waveform, sr=sample_rate)), ref=np.max)
    img = librosa.display.specshow(cqt_db, sr=sample_rate, x_axis='time', y_axis='cqt_hz', ax=axs[2, 0], cmap=cmap)
    axs[2, 0].set_title('Constant-Q Transform (CQT)')
    fig.colorbar(img, ax=axs[2, 0], format='%+2.0f dB')

    # Plot Tempogram
    tempo = librosa.feature.tempogram(y=waveform, sr=sample_rate)
    img = librosa.display.specshow(tempo, sr=sample_rate, x_axis='time', ax=axs[2, 1], cmap=cmap)
    axs[2, 1].set_title('Tempogram')
    fig.colorbar(img, ax=axs[2, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot to a bytes buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    # Show the plot
    plt.show()

    return image_base64

def predict_and_plot(file_path):
    """Predict the category of a sound and return base64-encoded image of plots."""
    waveform, sample_rate = load_wav_16k_mono(file_path)
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    # Predict category using YAMNet
    scores, embeddings, spectrogram = yamnet_model(waveform)
    top_class = tf.argmax(scores, axis=1).numpy()
    top_class_name = class_names[top_class[0]]

    # Print the predicted class
    print(f"Predicted Class: {top_class_name}")

    # Generate the feature plot
    image_base64 = plot_audio_features(waveform.numpy(), sample_rate)

    return top_class_name, image_base64

def process_audio(file_path, model='medium'):
    """Transcribe and translate audio using Whisper and Google Translator."""
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

# Main Script
if __name__ == "__main__":
    # Input video and audio paths
    video_path = 'bollywood-dialogues.mp4'
    audio_path = 'output_audio.mp3'
    # Extract audio from video
    extract_audio(video_path, audio_path)

    # Convert extracted audio to WAV format
    wav_file_path = convert_to_wav(audio_path)

    # Predict sound category and generate feature plot
    predicted_class, feature_plot = predict_and_plot(wav_file_path)

    # Transcribe and translate the audio
    language, transcription, translated_text = process_audio(audio_path)
