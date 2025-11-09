FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gating_service.py .

EXPOSE 8080

CMD ["python", "gating_service.py"]
