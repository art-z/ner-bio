FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

# gunicorn с uvicorn-воркерами
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]
