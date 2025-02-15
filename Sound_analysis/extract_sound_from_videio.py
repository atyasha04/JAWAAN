import subprocess

def extract_audio(video_path, audio_path):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path]
    subprocess.run(command, check=True)

video_path = '/content/videotoaudio.mp4'
audio_path = 'output_audio.mp3'

extract_audio(video_path, audio_path)