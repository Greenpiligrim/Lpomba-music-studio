import numpy as np
import os

def load_and_split_spectrograms(folder_path):
    """
    Load spectrograms from .npy files in the specified folder,
    split each spectrogram into two halves (X_train and y_train),
    and save the resulting arrays as new .npy files.

    Parameters:
        folder_path (str): Path to the folder containing .npy files.
    """
    # Get list of .npy files in the folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    # Initialize empty lists for X_train and y_train
    X_train = []
    y_train = []

    # Load data from .npy files
    for file in file_list:
        spectrogram = np.load(os.path.join(folder_path, file))

        # Split spectrogram into X_train and y_train (example: X_train is the first half, y_train is the second half)
        X_train.append(spectrogram[:, :spectrogram.shape[1]//2])
        y_train.append(spectrogram[:, spectrogram.shape[1]//2:])

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Save X_train and y_train as new .npy files
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)

    # Print dimensions of X_train and y_train
    print("Размер X_train:", X_train.shape)
    print("Размер y_train:", y_train.shape)

# # Пример использования функции:
# folder_path = '/Users/vladimirvasilenko/Documents/Visual Studio/Proj1/spectrograms'
# load_and_split_spectrograms(folder_path)
