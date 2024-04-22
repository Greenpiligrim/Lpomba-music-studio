import os
import librosa
import soundfile as sf

def split_audio_segments(audio_file, output_folder, segment_length=5, overlap=2):
    """
    Split an audio file into segments and save each segment as a separate file.

    Parameters:
        audio_file (str): Path to the audio file.
        output_folder (str): Path to the folder where segments will be saved.
        segment_length (float): Length of each segment in seconds (default is 5 seconds).
        overlap (float): Overlap between segments in seconds (default is 2 seconds).
    """
    try:
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=None)
    except Exception as e:
        print("Ошибка загрузки аудиофайла:", e)
        return

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Calculate total number of segments
    total_segments = int((len(audio) - segment_length * sr) / (segment_length * sr - overlap * sr)) + 1

    # Create and save each segment
    for i in range(total_segments):
        start = i * (segment_length - overlap) * sr
        end = start + segment_length * sr
        segment = audio[start:end]

        # Save segment as a separate file
        output_file = os.path.join(output_folder, f'segment_{i+1}.wav')
        sf.write(output_file, segment, sr)

        print("Сохранен файл:", output_file)

# Пример использования функции:
# audio_file = '/Users/vladimirvasilenko/Documents/Visual Studio/Proj1/trek.wav'
# output_folder = '/path/to/output/folder'
# split_audio_segments(audio_file, output_folder)
