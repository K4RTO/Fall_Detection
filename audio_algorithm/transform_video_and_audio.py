from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip


def extract_audio(video_path: str, audio_path: str):
    # 加载视频文件
    try:
        video = VideoFileClip(video_path)
    except Exception as e:
        print(f"Error: Failed to load video '{video_path}'. {e}")
        return None

    # 从视频中提取音频
    audio = video.audio
    if audio is None:
        print(f"{video_path} is without audio!")
        video.close()  # 记得关闭视频文件释放资源
        return None
    try:
        audio.write_audiofile(audio_path)
    except EOFError:
        print(f"current file does not exist: {audio_path}")

    video.close()


def add_audio_to_video(video_path: str, audio_path: str, output_path: str):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # 将音频设置到视频剪辑中
    video = video.set_audio(audio)

    # 输出最终的视频文件
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # 释放资源
    video.close()
    audio.close()


