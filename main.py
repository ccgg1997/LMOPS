import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import boto3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from datetime import datetime
import os



def main():
    local_file_path = 'kc_house_data.csv'
    model_filename = 'ml_model_regression.pkl'
    

    ##PASO 1:DESCARGAR LSO DATOS
    file_key = download_recent_data_file(local_file_path)
    if not file_key:
        print("No hay información nueva para procesar.")
        return
    

    ##2:REPROSESAMIENTRO DE DATOS
    X, y = preprocess_data(local_file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


    ##3:  Verificar si el mejor modelo ya existe
    if check_model_exists_in_s3(model_filename, folder="model/bestmodel"):
        s3_client = boto3.client('s3')
        bucket_name = 'datamlops'
        s3_client.download_file(bucket_name, f'bestmodel/{model_filename}', model_filename)
        with open(model_filename, 'rb') as file:
            regressor_best = pickle.load(file)
    else:
        regressor_best = None

    ##ENTRENAR NUEVO MODEL
    regressor_new = train_model(X_train, y_train)


    #evaluar ambos modelos
    y_pred_best = regressor_best.predict(X_test) if regressor_best else np.zeros(len(y_test))
    y_pred_new = regressor_new.predict(X_test)


    mae_best, mse_best, rmse_best = evaluate_model(y_test, y_pred_best)
    mae_new, mse_new, rmse_new = evaluate_model(y_test, y_pred_new)

    print(f"Mejor modelo: MAE={mae_best}, MSE={mse_best}, RMSE={rmse_best}")
    print(f"Nuevo modelo: MAE={mae_new}, MSE={mse_new}, RMSE={rmse_new}")
    

    if rmse_new < rmse_best:
        print("El nuevo modelo es mejor. Subiendo a S3 como 'bestmodel' y haciendo backup del anterior...")
        
        # Guardar el modelo nuevo como "bestmodel"
        with open(model_filename, 'wb') as file:
            pickle.dump(regressor_new, file)

        # Subir el nuevo modelo a S3 como "bestmodel"
        upload_model_to_s3(model_filename, folder="model/bestmodel")
        
        # Hacer un backup del modelo anterior si existía
        if regressor_best:
            backup_filename = f'backup_models/{model_filename}'
            upload_model_to_s3(model_filename, folder="backup_models")
        
    else:
        print("El modelo anterior es mejor. No se realiza ningún cambio.")



if __name__ == '__main__':
    main()

def check_model_exists_in_s3(model_filename, folder="model/bestmodel" ):
    s3_client = boto3.client('s3')
    bucket_name = 'datalomps'
    model_key = f'{folder}/{model_filename}'

    try:
        # Verificar si el modelo existe en el bucket S3
        s3_client.head_object(Bucket=bucket_name, Key=model_key)
        return True
    except Exception as e:
        print(f"Error al verificar el modelo en S3: {e}")
        return False

def upload_model_to_s3(model_filename, folder="model/bestmodel"):
    s3_client = boto3.client('s3')
    bucket_name = 'datamlops'
    model_key = f'{folder}/{model_filename}'

    try:
        # Subir el modelo entrenado al bucket S3
        s3_client.upload_file(model_filename, bucket_name, model_key)
        print(f"Modelo {model_filename} subido exitosamente a S3 en la carpeta 'model/'")
    except Exception as e:
        print(f"Error al subir el modelo a S3: {str(e)}")

def download_recent_data_file(local_file_path):
    s3_client = boto3.client('s3')
    bucket_name = 'datamlops'

    #listar los archivos de un bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix= 'data/')
    print(f"Archivos en el bucket: {response}")

    #ordenar los archivos por fecha, y solo sacar el nombre del archivo mas reciente
    try:
        files = response['Contents']
        # Ordenar por fecha
        files.sort(key=lambda x: x['LastModified'], reverse=True)
        file_key = files[0]['Key']
        print(f"Archivo más reciente: {file_key}")
        
        # Descargar archivo más reciente
        s3_client.download_file(bucket_name, file_key, local_file_path)
        print(f"Archivo descargado desde S3: {file_key}")
        return file_key
    except KeyError:
        print("No hay archivos disponibles en el bucket.")
        return False
    

###ENTRENAMIENTO DEL MODELO
def preprocess_data(file_path):
     # Leer datos
    data = pd.read_csv(file_path)
    print("Primeras filas del DataFrame:")
    print(data.head(5))

    # Eliminar columnas innecesarias
    data = data.drop(['date', 'id', 'zipcode'], axis=1, errors='ignore')

    # Separar características y etiquetas
    X = data.drop('price', axis=1).values
    y = data['price'].values
    return X, y
    
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_test, y_pred):
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse