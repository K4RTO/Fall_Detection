from audio_algorithm import audio_preprocess, spectrogram, model_apply

__all__ = ['fall_detection_by_audio']


def fall_detection_by_audio(audio_path: str, model_path: str) -> list(list()):
    """This will generate a map, the keys are fall time (seconds)"""

    # cut audio into slices
    audio_slices = audio_preprocess.split_audio(audio_path)

    # transfer audio slices to spectrogram images
    audio_slice_spectrogram_images = spectrogram.generate_spectrogram(audio_slices)

    # input spectrogram images of audio slices into the model and get the result
    result = model_apply.result_generate(model_path, audio_slice_spectrogram_images)

    return result
