### Happywhale - Whale and Dolphin Identification

#### 1. Описание задачи
Ссылка на Kaggle соревнование: https://www.kaggle.com/competitions/happy-whale-and-dolphin/overview

Необходимо реализовать модель, способную распознавать китов и дельфинов по их уникальным особенностям.

#### 2. Данные
Ссылка на данные: https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/315524

Здесь представленно несколько предобработанных датасетов, в которых намного меньше шума, чем в оригинальных сырых данных.

Всего представленно около 51 тыс. изображений разделенных на 15587 классов уникальных дельфинов и китов.

Для решения использовались датасеты Yolov5(Backfin) и Yolov5(FullBody).

```angular2html
- data
    - backfin_train.csv - данные о дельфинах и китах, где видно их плавники
    - df_train.csv - все данные
```

#### 3. Конфигурация

Файлы с конфигурацией:
1) src/configs/train_run_cinfig.yaml - общая конфигурация запуска
2) src/configs/data_configs - конфигурации для загрузки данных
3) src/configs/model_configs - конфигурации для моделей
4) src/configs/train_configs - конфигурации для параметров обучения

Для решения используем модель  Dolg-EfficientNet:
https://paperswithcode.com/paper/dolg-single-stage-image-retrieval-with-deep

#### 4. Запуск обучния

```angular2html
python3 src/train/train.py
```
