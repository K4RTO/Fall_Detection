import numpy as np
import librosa.feature
from PIL import Image
from pydub import AudioSegment


def get_mel_spectrogram_db(audio_segment: AudioSegment, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20,
                           fmax=8300, top_db=80):
    # 从AudioSegment对象提取音频样本
    samples = np.array(audio_segment.get_array_of_samples())

    # 如果音频是双声道的，取均值以转换为单声道
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)

    # 将样本数据类型转换为float32
    samples = samples.astype(np.float32, order='C') / 2**15

    # 重采样（如果需要）
    if audio_segment.frame_rate != sr:
        samples = librosa.resample(samples, orig_sr=audio_segment.frame_rate, target_sr=sr)

    max_len = int(3 * sr)
    if len(samples) < max_len:
        pad_left = int((max_len - len(samples)) // 2)
        pad_right = max_len - len(samples) - pad_left
        samples = np.pad(samples, (pad_left, pad_right), mode='constant', constant_values=(0, 0))
    else:
        samples = samples[:max_len]

    # 生成梅尔频谱图
    spec = librosa.feature.melspectrogram(y=samples, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def generate_spectrogram(audio_slices):
    audio_slice_images = []
    for audio_slice in audio_slices:
        audio_slice_spectrogram = get_mel_spectrogram_db(audio_slice, sr=44100, n_mels=224)
        audio_slice_img = spec_to_image(audio_slice_spectrogram)
        audio_slice_img = Image.fromarray(audio_slice_img[:, :224])  # same scale
        audio_slice_images.append(audio_slice_img)
    return audio_slice_images
