FROM python:3.11


RUN apt-get update && apt-get install -y gcc build-essential

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt



COPY data/ ./data/
COPY .env .
COPY api/app.py app.py
COPY api/model.py model.py

CMD ["gunicorn", "--bind", "0.0.0.0:8899", "app:app"]
