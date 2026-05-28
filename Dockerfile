FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_coach_interface/ .
COPY scripts/firestore_manager.py .

EXPOSE 8080

CMD ["streamlit", "run", "coach_app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
