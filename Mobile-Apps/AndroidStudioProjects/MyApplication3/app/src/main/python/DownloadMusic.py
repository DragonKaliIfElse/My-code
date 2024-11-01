from pytube import YouTube
from pydub import AudioSegment
import os
from android.os import Environment
from android.provider import MediaStore
from java.io import FileOutputStream
from java.io import OutputStream

def download_music(youtube_url,type_video):
    if type_video == "music":
        try:
            yt = YouTube(youtube_url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            download_dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath()
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            file_name = f"{yt.title}.mp3"
            audio_file_path = audio_stream.download(output_path=download_dir, filename=file_name)
 
            # Convert to MP3 using pydub
            audio = AudioSegment.from_file(audio_file_path)
            mp3_file_path = audio_file_path.replace(".mp4", ".mp3")
            audio.export(mp3_file_path, format="mp3")
            return audio_file_path
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
