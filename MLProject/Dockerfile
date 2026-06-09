FROM python:3.10-slim

#folder di dalam Docker
WORKDIR /app

#pindahkan semua file kodingan ke dalam Docker
COPY . /app

#install library yang dibutuhkan
RUN pip install pandas scikit-learn mlflow numpy joblib

#perintah otomatis yang dijalankan saat Docker dihidupkan
CMD ["python", "modelling.py"]