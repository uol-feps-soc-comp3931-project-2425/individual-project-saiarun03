# from pytube import YouTube
import subprocess
import cv2
import os

URL = 'https://www.youtube.com/shorts/Nf6iO1-aP7A'

# yt = YouTube(URL)
# video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
# video.download(filename='youtube_video.mp4')

# Download the video using yt-dlp
subprocess.run([
    'yt-dlp',
    '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    '-o', 'youtube_video.mp4',
    URL
])

