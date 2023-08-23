import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

class AudioProcessor():
    @staticmethod
    def open_file(file_path):
        time_series, sample_rate = librosa.load(file_path)
        return (time_series, sample_rate)

    @staticmethod
    def short_time_fourier_transform(time_series):
        return librosa.stft(time_series, hop_length=512, n_fft=2048*4)
    
    @staticmethod
    def create_mel_spectogram(time_series, sample_rate):
        time_series = AudioProcessor.pad_truncate(time_series, sample_rate, 10.0)
        spectogram_mag, _ = librosa.magphase(AudioProcessor.short_time_fourier_transform(time_series))
        mel_scale_spectogram = librosa.feature.melspectrogram(S=spectogram_mag, sr=sample_rate)
        return librosa.amplitude_to_db(mel_scale_spectogram, ref=np.min)
    
    @staticmethod
    def get_mel_spectogram(file_path: str):
        return AudioProcessor.create_mel_spectogram(*AudioProcessor.open_file(file_path))
    
    @staticmethod
    def plot_mel_spectogram(mel_spectogram: np.ndarray):
        librosa.display.specshow(mel_spectogram, x_axis="time", y_axis="mel", sr=22050)
        plt.colorbar(format="%+2.f")
        plt.show()

    @staticmethod
    def show_mel_spectogram(file_path: str):
        AudioProcessor.plot_mel_spectogram(AudioProcessor.get_mel_spectogram(file_path))

    @staticmethod
    def get_sample_rate(file_path: str):
        return AudioProcessor.open_file(file_path)[1]
    
    @staticmethod
    def get_time_series(file_path: str):
        return AudioProcessor.open_file(file_path)[0]
    
    @staticmethod
    def get_audio_file_path(file_path: str):
        if (file_path) and not (file_path.isspace()):
            file_path = os.path.join(os.getcwd(), "dataset", file_path.lstrip("/"))
            if os.path.isfile(file_path):
                return file_path
            else:
                raise Exception("This path doesn't point to a valid file")
        else:
            raise Exception("File path is empty or doesn't exist")
    
    @staticmethod
    def pad_truncate(time_series: np.ndarray, sample_rate: int, duration: float):
        if len(time_series) > sample_rate * duration:
            return time_series[:int(sample_rate * duration)]
        else:
            return np.hstack((time_series, np.zeros(int(sample_rate * duration) - len(time_series))))