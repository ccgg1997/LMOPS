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




s3_client = boto3.client('s3') #CLIENTE DE S3


bucket_name = 'datamlops'
file_key = 'datahouse.csv'
local_file_path = 'kc_house_data.csv'


s3_client.download_file(bucket_name, file_key, local_file_path)#DESCARGA DESDE S3
print(f"Archivo descargado desde S3: {file_key}")



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

# Multiple Liner Regression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)

#predicting the test set result
y_pred = regressor.predict(X_test)

#VISUALIZAR DATOS RESUDIALES
#fig = plt.figure(figsize=(10,5))
#residuals = (y_test- y_pred)
#fig = plt.figure(figsize=(10, 5))
#sns.histplot(residuals, kde=True, color='blue', bins=30)
#plt.title("Distribución de los Residuales")
#plt.xlabel("Residual")
#plt.ylabel("Frecuencia")
#plt.show()


#compare actual output values with predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(10)
print(df1)


#EVALUACIÓN DEL RENDIMIENTO
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:', metrics.explained_variance_score(y_test, y_pred))