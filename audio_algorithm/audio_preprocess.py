from typing import List

from pydub import AudioSegment


def split_audio(audio_path: str, window_length_ms=3000, slide_step_ms=1000) -> List[AudioSegment]:
    # 加载音频文件
    audio = AudioSegment.from_file(audio_path)

    audio_slices = []
    # 使用滑动窗口截取音频
    start = 0  # 开始时间
    while start + window_length_ms <= len(audio):
        segment = audio[start:start + window_length_ms]
        audio_slices.append(segment)
        start += slide_step_ms  # 移动窗口
    return audio_slices


