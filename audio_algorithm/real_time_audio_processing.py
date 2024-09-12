import os
from PIL import Image
from datetime import datetime
import pyaudio
import numpy as np
from collections import deque
import time
from spectrogram import generate_spectrogram, get_mel_spectrogram_db, spec_to_image
from pydub import AudioSegment
from model_apply import result_generate
# 音频设置
FORMAT = pyaudio.paInt16  # 16位整型
CHANNELS = 1  # 单声道
RATE = 44100  # 采样率
CHUNK = int(RATE / 10)  # 每次读取的帧数 (0.1秒的音频)
WINDOW_SIZE = 3  # 滑动窗口大小（秒）
SLIDE_INTERVAL = 1  # 滑动间隔（秒）
# 创建images文件夹（如果不存在）
output_dir = 'images'
# 初始化PyAudio
audio = pyaudio.PyAudio()

# 打开音频流
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# 使用deque存储滑动窗口的数据
window = deque(maxlen=int(RATE * WINDOW_SIZE / CHUNK))

# 记录第一个片段的开始时间
start_time = time.time()


def numpy_to_audio_segment(numpy_array, sample_rate):
    # 将numpy数组转换为字节数据
    byte_data = numpy_array.astype(np.int16).tobytes()

    # 创建AudioSegment对象
    audio_segment = AudioSegment(
        data=byte_data,
        sample_width=2,  # 16-bit audio
        frame_rate=sample_rate,
        channels=1  # 单声道
    )
    return audio_segment

def format_timestamp(timestamp):
    # 将时间戳转换为datetime对象
    dt = datetime.fromtimestamp(timestamp)
    # 格式化为当前年月日小时格式
    return dt.strftime('%Y-%m-%d_%H-%M-%S')

def process_audio_segment(segment, segment_start_time, segment_end_time):
    audio_segment = numpy_to_audio_segment(segment, RATE)
    # 处理音频片段的函数
    # 格式化时间戳
    start_time_str = format_timestamp(segment_start_time)
    end_time_str = format_timestamp(segment_end_time)

    # 打印处理的时间段
    print(f"Processing segment from {start_time_str} to {end_time_str}")
    # segment是numpy数组，包含两秒音频数据

    audio_slice_images = []

    audio_slice_spectrogram = get_mel_spectrogram_db(audio_segment, sr=44100, n_mels=224)
    audio_slice_img = spec_to_image(audio_slice_spectrogram)
    audio_slice_img = Image.fromarray(audio_slice_img[:, :224])  # same scale
    audio_slice_images.append(audio_slice_img)

    # # 保存每张频谱图像到images文件夹
    # for i, img in enumerate(audio_slice_images):
    #     img_filename = os.path.join(output_dir,
    #                                 f'spectrogram_{segment_start_time:.2f}_to_{segment_end_time:.2f}_{i}.png')
    #     img.save(img_filename)
    #     print(f"Saved {img_filename}")
    result = result_generate("../fall_detection_model_by_audio.pth", audio_slice_images)
    print(result)

try:
    while True:
        # 读取音频数据并转换为numpy数组
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # 添加到滑动窗口
        window.append(audio_data)

        # 计算当前片段的开始和结束时间
        current_time = time.time()
        segment_end_time = current_time
        segment_start_time = segment_end_time - WINDOW_SIZE

        # 每隔SLIDE_INTERVAL秒处理一次窗口数据
        if len(window) == window.maxlen:
            combined_data = np.concatenate(list(window))

            process_audio_segment(combined_data, segment_start_time, segment_end_time)
            time.sleep(SLIDE_INTERVAL)
finally:
    # 关闭流和PyAudio
    stream.stop_stream()
    stream.close()
    audio.terminate()
