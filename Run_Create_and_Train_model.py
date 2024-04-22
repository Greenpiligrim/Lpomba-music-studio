import numpy as np
from tensorflow import keras
from keras import layers
import soundfile as sf
import matplotlib.pyplot as plt
from Split_Audio import split_audio_segments
from Create_Spectrograms import batch_create_spectrograms
from Convert_Splits_to_numpy import load_and_split_spectrograms
import tensorflow as tf


# Нарезка Аудиофайла Нужно прописать путь к файлу и папке куда скидывать
audio_file = '/Users/vladimirvasilenko/Documents/Visual Studio/Lpomba-music-studio/trek.wav'
audio_folder = '/Users/vladimirvasilenko/Documents/Visual Studio/Lpomba-music-studio/splited_audio'
split_audio_segments(audio_file, audio_folder)

# Получение спектрограмм из нарезок
spectrogram_folder = '/Users/vladimirvasilenko/Documents/Visual Studio/Lpomba-music-studio/spectrogram_folder'
batch_create_spectrograms(audio_folder, spectrogram_folder)


#  Получение X_train and y_train
load_and_split_spectrograms(spectrogram_folder)

# Путь папки где  X_train and y_train Нужно прописать
# Load data (сокращаем данные)
X_train = np.load('/Users/vladimirvasilenko/Documents/Visual Studio/Lpomba-music-studio/X_train.npy')[:10]
y_train = np.load('/Users/vladimirvasilenko/Documents/Visual Studio/Lpomba-music-studio/y_train.npy')[:10]

# Add a zero column to X_train
X_train = np.concatenate([X_train, np.zeros((X_train.shape[0], X_train.shape[1], 1))], axis=-1)

# Create the model
model = keras.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.LSTM(128, return_sequences=True),
    layers.TimeDistributed(layers.Dense(X_train.shape[2], activation='sigmoid'))
])

# Compile the model with MSE loss and Adam optimizer
model.compile(loss='mse', optimizer='adam', run_eagerly=True)

# Print model summary
model.summary()

# Using GPU with Metal support for M1
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], 
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Train the model (меняем размер данных)
history = model.fit(X_train, y_train, batch_size=32, epochs=2, validation_split=0.2)

# Save the model in native Keras format
model.save('my_model.keras')  # Saving in native Keras format

# Generate audio (для одной минуты)
latent_dim = 128
random_latent_vector = np.random.normal(size=(1, X_train.shape[1], X_train.shape[2]))

# Определяем количество шагов для одной минуты (60 секунд * 22050 сэмплов в секунде)
steps_per_minute = 60 * 22050
generated_audio = []

for _ in range(steps_per_minute):
    predicted_steps = model.predict(random_latent_vector)
    generated_audio.extend(predicted_steps[0][-1])

    # Обновляем latent_vector для следующего шага
    random_latent_vector = np.concatenate([random_latent_vector[:, 1:], predicted_steps[:, -1:]], axis=1)

# Reshape and normalize generated audio
generated_audio = np.array(generated_audio).reshape(-1)
generated_audio /= np.max(np.abs(generated_audio))

# Сокращаем до одной минуты
generated_audio = generated_audio[:steps_per_minute]

# Save generated audio
output_file = '/Users/vladimirvasilenko/Documents/Visual Studio/Proj1/generated_audio_one_minute.wav'
sf.write(output_file, generated_audio, 22050, 'PCM_24')  # Save as WAV

# Plot the generated audio
plt.plot(generated_audio)
plt.title('Generated Audio')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()
