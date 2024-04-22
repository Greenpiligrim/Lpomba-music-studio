<img src="/baner.png" alt="" />

# Lpomba-music-studio

Deep Learning model

# Инструкции по установке

Cклонировать репозиторий. открыть Visual Studio -> Клонировать + ссылка:

```bash
git clone https://github.com/Greenpiligrim/Lpomba-music-studio.git
```

# Инструкции по установке библиотек из requirements txt

создаем виртуальное окружение:

```bash
python3 -m venv venv
```

активируем виртуальное окружение (для MacOS/Linux):

```bash
source venv/bin/activate
```

устанавливаем библиотеки из requirements.txt:

```bash
pip install -r requirements.txt
```

# Важно

Обязательно добавить аудио-трек на котором будет обучаться модель в проект. Создать две папки: одна для нарезки аудиофайла другая для спектрограмм.
В файле Run_Create_and_Train_model.py заменить все пути к созданным папкам и к аудиотреку

# Как пользоваться

Запустить Run_Create_and_Train_model.py

# Дополнительно

Добавил утилитку в файле utilite.py - для очистки, удаления лишних файлов в папках с нарезками - прост пропиши путь и количество сколько оставить
