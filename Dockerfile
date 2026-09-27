# Same Python version as the "at2" conda env used to train and save the
# models, so the pickled/joblib objects load correctly.
FROM python:3.11.16-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer when only the
# app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/

# Render sets $PORT at runtime; default to 8000 for a local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
