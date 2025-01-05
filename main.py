import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import boto3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from datetime import datetime
import os



def download_recent_data_file(local_file_path):
    s3_client = boto3.client('s3')
    bucket_name = 'datamlops'

    #listar los archivos de un bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix= 'data/')
    print(f"Archivos en el bucket: {response}")

    #ordenar los archivos por fecha, y solo sacar el nombre del archivo mas reciente
    try:
        files = response['Contents']
    except KeyError:
        return False
    
    #ordenar los archivos por fecha
    files.sort(key=lambda x: x['LastModified'], reverse=True)
    file_key = files[0]['Key']
    print(f"Archivo más reciente: {file_key}")
    
    if file_key == '' or file_key == None:
        return False

    s3_client.download_file(bucket_name, file_key, local_file_path)#DESCARGA DESDE S3
    print(f"Archivo descargado desde S3: {file_key}")
    
    


def main():
    local_file_path = 'kc_house_data.csv'
    file_key = download_recent_data_file('kc_house_data.csv')
    
    if file_key == False:
        print("No hay informacion nueva para reentrenar el modelo")
        return 0

    Data = pd.read_csv(local_file_path)#LEER LOS DATOS
    print("primeras filas del DataFrame:")
    print(Data.head(5))

    Data = Data.drop(['date', 'id', 'zipcode'], axis=1, errors='ignore')


    X = Data.drop('price',axis =1).values
    y = Data['price'].values


    print("Etiquetas (y):")
    print(y)

    #DIVIDIR LOS DATOS EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


    # Verificar si el mejor modelo ya existe
    model_filename = 'ml_model_regression.pkl'

    if check_model_exists_in_s3(model_filename, folder="bestModel"):
        print(f"Modelo {model_filename} en el bucket S3, no es necesario reentrenar")

        # Si el modelo existe, carga desde S3
        s3_client = boto3.client('s3')
        bucket_name = 'datalomps'
        s3_client.download_file(bucket_name, f'bestModel/{model_filename}', model_filename)
        
        
        with open(model_filename, 'rb') as file:
            regressor_best = pickle.load(file)

        print(f"Modelo {model_filename} cargado desde S3 (bestmodel)")

    else:
        print(f"Modelo {model_filename} no encontrado en el bucket S3, entrenando un nuevo modelo")
        regressor_best = None


    regressor_new = LinearRegression()
    regressor_new.fit(X_train, y_train)


    #evaluar ambos modelos
    y_pred_best = regressor_best.predict(X_test) if regressor_best else np.array([0] * len(y_test))
    y_pred_new = regressor_new.predict(X_test)



    #compare actual output values with predicted values
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(10)
    print(df1)


    #EVALUACIÓN DEL RENDIMIENTO bestmodel
    print("Evaluación del modelo mejor (bestmodel):")
    mae_best = metrics.mean_absolute_error(y_test, y_pred_best)
    mse_best = metrics.mean_squared_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mse_best)
    print(f"MAE: {mae_best}, MSE: {mse_best}, RMSE: {rmse_best}")

    print("Evaluación del modelo nuevo:")
    mae_new = metrics.mean_absolute_error(y_test, y_pred_new)
    mse_new = metrics.mean_squared_error(y_test, y_pred_new)
    rmse_new = np.sqrt(mse_new)
    print(f"MAE: {mae_new}, MSE: {mse_new}, RMSE: {rmse_new}")

    if rmse_new < rmse_best:
        print("El nuevo modelo es mejor. Subiendo a S3 como 'bestmodel' y haciendo backup del anterior...")
        
        # Guardar el modelo nuevo como "bestmodel"
        with open(model_filename, 'wb') as file:
            pickle.dump(regressor_new, file)

        # Subir el nuevo modelo a S3 como "bestmodel"
        upload_model_to_s3(model_filename, folder="bestmodel")
        
        # Hacer un backup del modelo anterior si existía
        if regressor_best:
            backup_filename = f'backup_models/{model_filename}'
            upload_model_to_s3(model_filename, folder="backup_models")
        
    else:
        print("El modelo anterior es mejor. No se realiza ningún cambio.")



    #print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    #print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    #print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    #print('VarScore:', metrics.explained_variance_score(y_test, y_pred))

if __name__ == '__main__':
    main()

def check_model_exists_in_s3(model_filename):
    s3_client = boto3.client('s3')
    bucket_name = 'datalomps'
    model_key = f'model/{model_filename}'

    try:
        # Verificar si el modelo existe en el bucket S3
        s3_client.head_object(Bucket=bucket_name, Key=model_key)
        return True
    except Exception as e:
        return False

def upload_model_to_s3(model_filename):
    s3_client = boto3.client('s3')
    bucket_name = 'datamlops'
    model_key = f'model/{model_filename}'

    try:
        # Subir el modelo entrenado al bucket S3
        s3_client.upload_file(model_filename, bucket_name, model_key)
        print(f"Modelo {model_filename} subido exitosamente a S3 en la carpeta 'model/'")
    except Exception as e:
        print(f"Error al subir el modelo a S3: {str(e)}")


