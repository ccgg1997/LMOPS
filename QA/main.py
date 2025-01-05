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
    local_file_path = 'kc_house_data_qa.csv'
    Data = pd.read_csv(local_file_path)#Read the data
    
    #Remove columns that are not needed
    Data = Data.drop(['date', 'id', 'zipcode'], axis=1, errors='ignore')

    #Split the data into feature data and target data
    X = Data.drop('price',axis =1).values
    y = Data['price'].values

    #Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

    regressor_new = LinearRegression()
    regressor_new.fit(X_train, y_train)

    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('VarScore:', metrics.explained_variance_score(y_test, y_pred))

if __name__ == '__main__':
    main()

