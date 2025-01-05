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


    # Verificar si el modelo ya existe
    model_filename = 'ml_model_regression.pkl'

    if check_model_exists_in_s3(model_filename, folder="bestModel"):
        print(f"Modelo {model_filename} en el bucket S3, no es necesario reentrenar")
        return 0
    # Si el modelo existe, cargarlo desde S3
        s3_client = boto3.client('s3')
        bucket_name = 'datalomps'
        s3_client.download_file(bucket_name, f'bestModel/{model_filename}', model_filename)
        
        
        with open(model_filename, 'rb') as file:
            regressor_best = pickle.load(file)

        print(f"Modelo {model_filename} cargado desde S3 (bestmodel)")

    else:
        print(f"Modelo {model_filename} no encontrado en el bucket S3, entrenando un nuevo modelo")
        
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

        with open(model_filename, 'wb') as file:
            pickle.dump(regressor, file)

        print(f"Modelo {model_filename} entrenado y guardado localmente")

        upload_model_to_s3(model_filename)

 

    #predicting the test set result
    y_pred = regressor.predict(X_test)


    #compare actual output values with predicted values
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(10)
    print(df1)


    #EVALUACIÓN DEL RENDIMIENTO
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('VarScore:', metrics.explained_variance_score(y_test, y_pred))

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


