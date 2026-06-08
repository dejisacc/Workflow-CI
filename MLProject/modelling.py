import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
import joblib

def train_model():
    # 1. Mengatur rute dinamis (Dynamic Path)
    # BASE_DIR saat ini adalah folder 'Membangun_model'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # directory dataset hasil preprocessing berada di folder 'Customer_shopping_preprocessing' yang sejajar dengan 'Membangun_model'
    data_path = os.path.abspath(os.path.join(BASE_DIR, "Customer_shopping_preprocessing", "shopping_trends_preprocessed.csv"))
        
    print(f"Membaca data hasil preprocessing dari:\n{data_path}\n")
    
    if not os.path.exists(data_path):
        print("Peringatan: File dataset tidak ditemukan.")
        return

    # Load dataset ke variabel df_clean
    df_clean = pd.read_csv(data_path)
    
    # 2. Pengaturan MLflow Tracking Menggunakan SQLite Backend
    db_path = os.path.join(BASE_DIR, "mlflow.db")
    
    # Mengonversi format slash agar dikenali oleh SQLAlchemy URI
    sqlite_uri = f"sqlite:///{db_path.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(sqlite_uri)
    
    # Menentukan nama eksperimen di database
    mlflow.set_experiment("Submission Membangun Sistem Machine Learning - KMeans Clustering")
    
    # 3. Mengaktifkan Autolog MLflow
    mlflow.sklearn.autolog(log_models=True)
    
    # 4. Melatih Model K-Means
    print("Memulai pencatatan ke MLflow Tracking UI (SQLite)...")
    with mlflow.start_run(run_name="KMeans_SQLite_Run") as run:
        n_clusters = 4
        print(f"Melatih model K-Means dengan {n_clusters} klaster...")
        
        model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        model.fit(df_clean)

        # Save model
        # Membuat path ke folder 'model' yang sejajar dengan modelling.py
        model_dir = os.path.abspath(os.path.join(BASE_DIR, "model"))
        
        # Pastikan folder 'model' ada (jika belum, buat otomatis)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Simpan model
        joblib.dump(model, os.path.join(model_dir, "kmeans_model.joblib"))
        print(f"\n[SUKSES] Model berhasil disimpan di: {os.path.join(model_dir, 'kmeans_model.joblib')}")
        # ----------------------------------------------
        
        run_id = run.info.run_id
        print(f"\n[SUKSES] Model berhasil dilatih! Run ID: {run_id}")
        print(f"Semua histori tracking berhasil dikunci di:\n{db_path}")

if __name__ == "__main__":
    train_model()