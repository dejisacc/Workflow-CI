import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
import joblib

def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    data_path = os.path.join(BASE_DIR, "shopping_trends_preprocessed.csv")
        
    print(f"Membaca data hasil preprocessing dari:\n{data_path}\n")
    
    if not os.path.exists(data_path):
        print("Peringatan: File dataset tidak ditemukan.")
        return

    df_clean = pd.read_csv(data_path)
    
    db_path = os.path.join(BASE_DIR, "mlflow.db")
    sqlite_uri = f"sqlite:///{db_path.replace(os.sep, '/')}"
    
    mlflow.set_tracking_uri(sqlite_uri)
    mlflow.set_experiment("Submission Membangun Sistem Machine Learning - KMeans Clustering")
    mlflow.sklearn.autolog(log_models=True)
    
    print("Memulai pencatatan ke MLflow Tracking UI (SQLite)...")
    with mlflow.start_run(run_name="KMeans_SQLite_Run") as run:
        n_clusters = 4
        print(f"Melatih model K-Means dengan {n_clusters} klaster...")
        
        model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        model.fit(df_clean)

        model_dir = os.path.join(BASE_DIR, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        joblib.dump(model, os.path.join(model_dir, "kmeans_model.joblib"))
        print(f"\n[SUKSES] Model berhasil disimpan di: {os.path.join(model_dir, 'kmeans_model.joblib')}")
    
        run_id = run.info.run_id
        print(f"\n[SUKSES] Model berhasil dilatih! Run ID: {run_id}")

if __name__ == "__main__":
    train_model()