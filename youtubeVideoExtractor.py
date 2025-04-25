# from pytube import YouTube
import subprocess

URL = 'https://www.youtube.com/watch?v=fC_tVugmn2w'


# Download the video using yt-dlp
subprocess.run([
    'yt-dlp',
    '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    '-o', 'youtube_video.mp4',
    URL
])

