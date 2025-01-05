import pickle
import numpy as np
from sklearn import metrics
import boto3

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
    if check_model_exists_in_s3(model_filename, bucket_name, folder="model/bestModel"):
        regressor_best = download_model_from_s3(model_filename, bucket_name, folder="model/bestModel")

    y_pred_best = regressor_best.predict(X_test) if regressor_best else np.zeros(len(y_test))
    y_pred_new = regressor_new.predict(X_test)

    mae_best, mse_best, rmse_best = evaluate_model(y_test, y_pred_best)
    mae_new, mse_new, rmse_new = evaluate_model(y_test, y_pred_new)

    print(f"Mejor modelo: MAE={mae_best}, MSE={mse_best}, RMSE={rmse_best}")
    print(f"Nuevo modelo: MAE={mae_new}, MSE={mse_new}, RMSE={rmse_new}")

    if rmse_new < rmse_best:
        print("El nuevo modelo es mejor. Subiendo a S3...")
        with open(model_filename, 'wb') as file:
            pickle.dump(regressor_new, file)
        upload_model_to_s3(model_filename, bucket_name, folder="model/bestModel")
    else:
        print("El modelo anterior es mejor. No se realiza ningún cambio.")

def upload_model_to_s3(model_filename, bucket_name, folder="model/bestModel"):
    s3_client = boto3.client('s3')
    model_key = f'{folder}/{model_filename}'
    s3_client.upload_file(model_filename, bucket_name, model_key)
    print(f"Modelo {model_filename} subido exitosamente a S3 en '{folder}'.")
