# удалялка утилита
# если мого файлов в какой папке чтоб вручную не удалять - задаешь кол-во сколько оставить - все остальное в гарбидж

import os

# Путь к папке с аудио файлами Нужно прописать
folder_path = '/Users/vladimirvasilenko/Documents/Visual Studio/Lpomba-music-studio/spectrogram_folder'

# Получаем список всех файлов в папке
files = os.listdir(folder_path)

# Если в папке есть более одного файла
if len(files) > 1:
    # Удаляем все файлы,кроме 
    for filename in files[1:]:
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"Удален файл: {filename}")
else:
    print("В папке нет файлов для удаления, кроме первого.")
