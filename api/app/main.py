import os
import time
import logging
import numpy as np
import onnxruntime as ort
import asyncio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
from starlette.middleware.base import BaseHTTPMiddleware


# --- логи ---
LOG_FILE = os.getenv("LOG_FILE", "requests.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "/model/model.onnx")
HF_URL = "https://huggingface.co/artzzzzz/ner/resolve/main/model.onnx"

def ensure_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Модель не найдена в {MODEL_PATH}, загружаем с Hugging Face")
        try:
            r = requests.get(HF_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Модель загружена")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
    else:
        print(f"Модель уже загружена в {MODEL_PATH}")

ensure_model()

TOKENIZER_NAME = "DeepPavlov/rubert-base-cased"
MAX_LEN = 500

LABELS = [
    "O",
    "B-TYPE", "I-TYPE",
    "B-BRAND", "I-BRAND",
    "B-VOLUME", "I-VOLUME",
    "B-PERCENT", "I-PERCENT"
]

# --- middleware для активных запросов ---
active_requests = 0
max_active_requests = 0
lock = asyncio.Lock()

async def track_requests_middleware(request: Request, call_next):
    global active_requests, max_active_requests

    async with lock:
        active_requests += 1
        if active_requests > max_active_requests:
            max_active_requests = active_requests
        current = active_requests
        peak = max_active_requests
    print(f"▶️ старт запроса, активных={current}, максимум={peak}")

    response = await call_next(request)

    async with lock:
        active_requests -= 1
        current = active_requests
    print(f"✅ конец запроса, активных={current}, максимум={peak}")

    return response

app.add_middleware(BaseHTTPMiddleware, dispatch=track_requests_middleware)

# --- загружаем токенизатор и модель ---
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=["CPUExecutionProvider"]
)

class UserQuery(BaseModel):
    input: str

def merge_bio_spans(spans):
    # --- объединение BIO-спанов в сущности ---
    if not spans:
        return []
    merged = []
    current_start, current_end, current_label = spans[0]
    for start, end, label in spans[1:]:
        if current_end == start:
            current_end = end
        else:
            merged.append({
                "start_index": current_start,
                "end_index": current_end,
                "entity": current_label
            })
            current_start, current_end, current_label = start, end, label
    merged.append({
        "start_index": current_start,
        "end_index": current_end,
        "entity": current_label
    })
    return merged

# --- очередь для батчинга ---
queue = asyncio.Queue()
BATCH_SIZE = 5
MAX_WAIT_MS = 50

async def batch_worker():
    while True:
        reqs = []
        # ожидаем первый запрос
        item = await queue.get()
        reqs.append(item)

        # собираем пачку
        try:
            while len(reqs) < BATCH_SIZE:
                item = await asyncio.wait_for(queue.get(), timeout=MAX_WAIT_MS/1000)
                reqs.append(item)
        except asyncio.TimeoutError:
            pass

        texts = [r["text"] for r in reqs]
        # токенизация сразу пачкой
        tokens = tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_offsets_mapping=True
        )
        offsets = tokens.pop("offset_mapping")

        input_names = {i.name for i in session.get_inputs()}
        ort_inputs = {k: v for k, v in tokens.items() if k in input_names}

        outputs = session.run(None, ort_inputs)
        logits = outputs[0]
        pred_ids_batch = np.argmax(logits, axis=-1)

        # раздаем результаты каждому запросу
        for i, r in enumerate(reqs):
            spans = []
            for idx, label_id in enumerate(pred_ids_batch[i]):
                start_char, end_char = offsets[i][idx]
                if start_char == end_char:
                    continue
                label = LABELS[label_id]
                spans.append((int(start_char), int(end_char), label))
            entities = merge_bio_spans(spans)
            r["future"].set_result(entities)

@app.on_event("startup")
async def startup():
    # прогрев модели
    dummy = tokenizer("warmup", return_tensors="np", truncation=True, padding=True, max_length=8)
    input_names = {i.name for i in session.get_inputs()}
    ort_inputs = {k: v for k, v in dummy.items() if k in input_names}
    session.run(None, ort_inputs)
    print("Warmup done")

    # запуск воркера батчинга
    asyncio.create_task(batch_worker())

@app.post("/predict")
async def predict(req: UserQuery, request: Request):
    input_text = (req.input or "").strip().lower()
    if not input_text:
        return []
    if len(input_text) > MAX_LEN:
        raise HTTPException(status_code=413, detail=f"Input too long (>{MAX_LEN} chars)")

    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await queue.put({"text": input_text, "future": fut})
    entities = await fut

    elapsed = 0
    client_ip = request.client.host if request.client else "unknown"
    logging.info(
        "Client %s | Input: %r | Entities: %s | Time: %.1f ms",
        client_ip, input_text, entities, elapsed
    )
    print(f"[PREDICT] {client_ip} | '{input_text}' | {elapsed:.1f} ms")
    return entities
