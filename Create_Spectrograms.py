import os
import librosa
import numpy as np

def batch_create_spectrograms(audio_folder, output_folder, n_fft=2048, hop_length=512):
    """
    Create spectrograms for all audio files in the given folder and save them as numpy files.

    Parameters:
        audio_folder (str): Path to the folder containing audio files.
        output_folder (str): Path to the folder where spectrograms will be saved.
        n_fft (int): Length of the FFT window (default is 2048).
        hop_length (int): Number of samples between successive frames (default is 512).
    """
    # Get list of audio files in the folder
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Function to create spectrogram from audio file
    def create_spectrogram(audio_file):
        try:
            audio, sr = librosa.load(audio_file, sr=None)
        except Exception as e:
            print(f"Ошибка загрузки аудиофайла {audio_file}: {e}")
            return None

        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram

    # Iterate through each audio file, create spectrogram, and save as numpy
    for audio_file in audio_files:
        input_path = os.path.join(audio_folder, audio_file)
        output_path = os.path.join(output_folder, os.path.splitext(audio_file)[0] + '.npy')

        spectrogram = create_spectrogram(input_path)

        if spectrogram is not None:
            np.save(output_path, spectrogram)
            print("Сохранен файл:", output_path)

# Пример использования функции:
# audio_folder = '/Users/vladimirvasilenko/Documents/Visual Studio/Proj1/treks'
# output_folder = '/Users/vladimirvasilenko/Documents/Visual Studio/Proj1/spectrograms'
# batch_create_spectrograms(audio_folder, output_folder)
