FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
COPY best_practices.txt .
COPY blockedphrases.txt .
COPY facilities.txt .
COPY app.py .
COPY controllers/ controllers/
COPY models/ models/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 55555

# By default, we’ll run Gunicorn to serve the Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:55555", "app:app"]