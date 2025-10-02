# NER для e-commerce запросов

Проект решает задачу **Named Entity Recognition (NER)** для русскоязычных поисковых запросов в e-commerce.
Поддерживаемые сущности (BIO-разметка):

- `TYPE` — тип продукта
- `BRAND` — бренд
- `VOLUME` — объём/вес
- `PERCENT` — проценты жирности/содержания

## Структура проекта
```bash
root/
│── main.py
│── data/
│── preprocess/
│   ├── __init__.py
│   ├── utils.py
│   ├── stopwords.py
│   ├── brands.py
│   ├── bio_fix.py
│   └── tokens.py
│
│── augment/
│   ├── __init__.py
│   ├── volumes.py
│   ├── adj.py
│   ├── noun_vol.py
│   ├── word_vol.py
│   └── extras.py
│
│── analysis/
│   ├── __init__.py
│   ├── stats.py
│   └── report.py
│
└── requirements.txt
 ``` 
 
## Установка

```bash
git clone https://github.com/art-z/ner-bio.git
cd ner-bio
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Генерация датасета
```bash
python main.py
```

## Собрать отчет / статистику train.csv
```bash
python -m analysis.report 
``` 
Результат: data/analysis_report.txt 
 
 ```bash
python -m analysis.stats
``` 
Результат: data/stats.csv, data/stats.json