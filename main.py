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



def download_recent_data_file(local_file_path):
    s3_client = boto3.client('s3')
    bucket_name = 'datamlops'

    #listar los archivos de un bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)
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

    try:
        # Intentar cargar el modelo entrenado previamente
        with open(model_filename, 'rb') as file:
            regressor = pickle.load(file)
            print("Modelo cargado desde el archivo")
    except FileNotFoundError:
        # Si el modelo no existe, entrenar uno nuevo
        print("Modelo no encontrado, entrenando un nuevo modelo")
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Guardar el modelo entrenado
        with open(model_filename, 'wb') as file:
            pickle.dump(regressor, file)
        print("Modelo entrenado y guardado")

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