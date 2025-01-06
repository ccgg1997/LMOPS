import pickle
import numpy as np
from sklearn import metrics
import boto3
from datetime import datetime

def evaluate_model(y_test, y_pred):
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def check_model_exists_in_s3(model_filename, bucket_name, folder="model/bestModel"):
    s3_client = boto3.client('s3')
    model_key = f'{folder}/{model_filename}'
    try:
        s3_client.head_object(Bucket=bucket_name, Key=model_key)
        return True
    except Exception:
        return False

def download_model_from_s3(model_filename, bucket_name, folder="model/bestModel"):
    s3_client = boto3.client('s3')
    model_key = f'{folder}/{model_filename}'
    local_file_path = model_filename
    s3_client.download_file(bucket_name, model_key, local_file_path)
    print(f"Modelo descargado desde S3: {local_file_path}")
    with open(local_file_path, 'rb') as file:
        return pickle.load(file)

def evaluate_and_compare_models(regressor_new, X_test, y_test, model_filename, bucket_name):
    regressor_best = None
    source_folder = "model/bestModel"
    backup_folder = "model/backup_models"

    # Verificar si ya existe un mejor modelo en S3
    if check_model_exists_in_s3(model_filename, bucket_name, folder=source_folder):
        regressor_best = download_model_from_s3(model_filename, bucket_name, folder=source_folder)

    # Realizar predicciones
    y_pred_best = regressor_best.predict(X_test) if regressor_best else np.zeros(len(y_test))
    y_pred_new = regressor_new.predict(X_test)

    # Evaluar modelos
    mae_best, mse_best, rmse_best = evaluate_model(y_test, y_pred_best)
    mae_new, mse_new, rmse_new = evaluate_model(y_test, y_pred_new)

    print(f"Mejor modelo: MAE={mae_best}, MSE={mse_best}, RMSE={rmse_best}")
    print(f"Nuevo modelo: MAE={mae_new}, MSE={mse_new}, RMSE={rmse_new}")

    # Comparar modelos
    if rmse_new < rmse_best:
        print("El nuevo modelo es mejor. Actualizando S3...")

        # Mover el modelo anterior a la carpeta de respaldo si existe
        if regressor_best:
            move_model_to_backup(model_filename, bucket_name, source_folder, backup_folder)

        # Guardar el nuevo modelo localmente
        with open(model_filename, 'wb') as file:
            pickle.dump(regressor_new, file)

        # Subir el nuevo modelo a S3 como el mejor modelo
        upload_model_to_s3(model_filename, bucket_name, folder=source_folder)
    else:
        print("El modelo anterior sigue siendo el mejor. No se realiza ningún cambio.")

def upload_model_to_s3(model_filename, bucket_name, folder="model/bestModel"):
    s3_client = boto3.client('s3')
    model_key = f'{folder}/{model_filename}'
    s3_client.upload_file(model_filename, bucket_name, model_key)
    print(f"Modelo {model_filename} subido exitosamente a S3 en '{folder}'.")

def move_model_to_backup(model_filename, bucket_name, source_folder, backup_folder):
    s3_client = boto3.client('s3')

    # Define las claves del modelo en el bucket
    source_key = f'{source_folder}/{model_filename}'
    #backup_key = f'{backup_folder}/{model_filename}'

    # Añadir fecha y hora al modelo de respaldo
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    backup_filename_with_date = f"{current_time}_{model_filename}"
    backup_key = f'{backup_folder}/{backup_filename_with_date}'

    try:
        # Copiar el modelo a la carpeta de respaldo
        copy_source = {'Bucket': bucket_name, 'Key': source_key}
        s3_client.copy_object(CopySource=copy_source, Bucket=bucket_name, Key=backup_key)
        print(f"Modelo copiado a la carpeta de respaldo: {backup_key}")

        # Eliminar el modelo de la carpeta original
        s3_client.delete_object(Bucket=bucket_name, Key=source_key)
        print(f"Modelo eliminado de la carpeta original: {source_folder}")
    except Exception as e:
        print(f"Error al mover el modelo a la carpeta de respaldo: {e}")
