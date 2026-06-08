# 1. Gunakan sistem operasi Linux dengan Python 3.10
FROM python:3.10-slim

# 2. Tentukan folder kerja di dalam Docker
WORKDIR /app

# 3. Pindahkan semua file kodingan dari laptop ke dalam Docker
COPY . /app

# 4. Install library yang dibutuhkan
RUN pip install pandas scikit-learn mlflow numpy joblib

# 5. Perintah otomatis yang dijalankan saat Docker dihidupkan
CMD ["python", "modelling.py"]