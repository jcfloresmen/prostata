FROM python:3.11-slim
ENV PORT=8000
COPY requirements.txt /
RUN pip install -r requirements.txt
COPY ./app /app
COPY modelo_vgg16.h5 /
COPY breeds.txt /

ENTRYPOINT uvicorn app.main:app --host 0.0.0.0 --port $PORT


