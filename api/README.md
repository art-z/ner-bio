# NER API (ONNX + FastAPI)

API-сервис для инференса модели распознавания сущностей (TYPE, BRAND, VOLUME, PERCENT)  
на основе RuBERT в формате ONNX.  
Модель заранее обучается и конвертируется в ONNX.

---

## Установка
```bash
docker compose up --build
``` 
Сервисы будут доступны на портах:
```bash
http://localhost:8088 
http://localhost:8089 
http://localhost:8090
``` 

## Эндпоинты

```bash
POST /predict
``` 
 
Запрос:
```bash
{
  "input": "вода питьевая 1 л"
}
``` 

Ответ:
```bash
[
  {"start_index": 0, "end_index": 4, "entity": "B-TYPE"},
  {"start_index": 5, "end_index": 13, "entity": "I-TYPE"},
  {"start_index": 14, "end_index": 15, "entity": "B-VOLUME"},
  {"start_index": 16, "end_index": 17, "entity": "I-VOLUME"}
] 
``` 

## Пример продакшн-запуска (3 реплики + балансировщик)
1. Поднять контейнеры:
```bash
docker-compose up -d
``` 
2. Настроить Nginx как балансировщик между qa-api-1..3
```bash
# --- API (балансировка) ---
upstream api_ballance {
    least_conn;  # балансировка по минимальному количеству соединений
    server 127.0.0.1:8088;
    server 127.0.0.1:8089;
    server 127.0.0.1:8090;
}
server { 
    location /api/ {
        proxy_pass api_ballance;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # CORS
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'Origin, Content-Type, Accept, Authorization' always;

        # preflight OPTIONS
        if ($request_method = OPTIONS) {
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS';
            add_header 'Access-Control-Allow-Headers' 'Origin, Content-Type, Accept, Authorization';
            add_header 'Content-Length' 0;
            add_header 'Content-Type' 'text/plain';
            return 204;
        }
    }
}    
    
``` 

API будет доступен по одному адресу
```bash
http://localhost:8080/api/predict
``` 