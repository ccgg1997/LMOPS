import boto3
import pandas as pd
from sklearn.model_selection import train_test_split


def download_recent_data_file(local_file_path, bucket_name):
    s3_client = boto3.client("s3")

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="data/")
    print(f"Archivos en el bucket: {response}")

    try:
        files = response["Contents"]
        files.sort(key=lambda x: x["LastModified"], reverse=True)
        file_key = files[0]["Key"]
        print(f"Archivo más reciente: {file_key}")

        s3_client.download_file(bucket_name, file_key, local_file_path)
        print(f"Archivo descargado desde S3: {file_key}")
        return file_key
    except KeyError:
        print("No hay archivos disponibles en el bucket.")
        return False


def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    print("Primeras filas del DataFrame:")
    print(data.head(5))

    data = data.drop(["date", "id", "zipcode"], axis=1, errors="ignore")
    X = data.drop("price", axis=1).values
    y = data["price"].values
    return X, y


def download_and_preprocess_data(local_file_path, bucket_name):
    file_key = download_recent_data_file(local_file_path, bucket_name)
    if not file_key:
        print("No hay datos nuevos para procesar.")
        return None, None, None, None

    X, y = preprocess_data(local_file_path)
    return train_test_split(X, y, test_size=0.33, random_state=101)


def split_data(local_file_path):
    X, y = preprocess_data(local_file_path)
    return train_test_split(X, y, test_size=0.33, random_state=101)


def upload_data_to_s3(data_filename, bucket_name, folder="data"):
    s3_client = boto3.client("s3")
    model_key = f"data/{data_filename}"
    s3_client.upload_file(data_filename, bucket_name, model_key)
    print(f"Modelo {data_filename} subido exitosamente a S3 en '{folder}'.")
