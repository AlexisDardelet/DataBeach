import yt_dlp

def download_video(url):
    ydl_opts = {
        'format': 'best',
        'outtmpl': '%(title)s.%(ext)s',
        'cookies'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    video_url = input("Enter video URL: ")
    print("Downloading...")
    download_video(video_url)
    print("Download complete!")