<img src="/baner.png" alt="" />

# Lpomba-music-studio

Deep Learning model

# Инструкции по установке

Cклонировать репозиторий на свой компьютер:

```bash
git clone https://github.com/Greenpiligrim/Lpomba-music-studio.git
```

# Инструкции по установке библиотек из requirements txt

```bash
python3 -m venv venv  # создаем виртуальное окружение
```

```bash
source venv/bin/activate  # активируем виртуальное окружение (для MacOS/Linux)
```

```bash
pip install -r requirements.txt  # устанавливаем библиотеки из requirements.txt
```

# Важно

Обязательно добавить аудио-трек на котором будет обучаться модель в проект. Создать две папки: одна для нарезки аудиофайла другая для спектрограмм.
В файле Run_Create_and_Train_model.py заменить все пути к созданным папкам и к аудиотреку

# Как пользоваться

Запустить Run_Create_and_Train_model.py

# Дополнительно

Добавил утилитку в файле utilite.py - для очистки, удаления лишних файлов в папках с нарезками - прост пропиши путь и количество сколько оставить
