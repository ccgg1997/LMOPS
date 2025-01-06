# app.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from pipeline.etl import download_and_preprocess_data, split_data
from pipeline.model_build import train_and_save_model
from pipeline.model_predict import evaluate_and_compare_models


# Función para cargar el modelo
def cargar_modelo(modelo_path="../model/ml_model_regression.pkl"):
    if os.path.exists(modelo_path):
        modelo = joblib.load(modelo_path)
        return modelo
    else:
        return None


# Cargar el modelo existente
modelo = cargar_modelo(modelo_path="../model/ml_model_regression.pkl")

st.title("Predicción de Precios de Casas (MLOPS3)")

# Crear pestañas
tab1, tab2 = st.tabs(["🔮 Predecir Precio", "📈 Actualizar Modelo"])

with tab1:
    st.header("Ingresar Datos para la Predicción")

    st.markdown(
        "Por favor, ingresa los siguientes parámetros (o usa los valores por defecto):"
    )

    # Parámetros requeridos por el modelo con valores por defecto
    date = st.text_input("Fecha de Venta (YYYY-MM-DD)", value="2023-01-01")
    bedrooms = st.number_input(
        "Número de Habitaciones", min_value=1, max_value=10, value=3
    )
    bathrooms = st.number_input("Número de Baños", min_value=1, max_value=10, value=2)
    sqft_living = st.number_input(
        "Metros Cuadrados Habitables", min_value=20, max_value=10000, value=1500
    )
    sqft_lot = st.number_input(
        "Tamaño del Lote (m²)", min_value=20, max_value=50000, value=5000
    )
    floors = st.number_input("Número de Pisos", min_value=1, max_value=5, value=1)
    waterfront = st.selectbox(
        "¿Tiene Vista al Agua?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No"
    )
    view = st.number_input("Vista (de 0 a 4)", min_value=0, max_value=4, value=0)
    condition = st.number_input(
        "Condición (de 1 a 5)", min_value=1, max_value=5, value=3
    )
    grade = st.number_input(
        "Calidad de Construcción (de 1 a 13)", min_value=1, max_value=13, value=7
    )
    sqft_above = st.number_input(
        "Metros Cuadrados sobre el Nivel del Suelo",
        min_value=20,
        max_value=10000,
        value=1200,
    )
    sqft_basement = st.number_input(
        "Metros Cuadrados en el Sótano", min_value=0, max_value=5000, value=300
    )
    yr_built = st.number_input(
        "Año de Construcción", min_value=1800, max_value=2025, value=1990
    )
    yr_renovated = st.number_input(
        "Año de Renovación (0 si no aplica)", min_value=0, max_value=2025, value=0
    )
    lat = st.number_input("Latitud", value=47.5112)
    long = st.number_input("Longitud", value=-122.257)
    sqft_living15 = st.number_input(
        "Promedio de Metros Habitables en Vecindario", value=1500
    )
    sqft_lot15 = st.number_input("Promedio de Tamaño de Lote en Vecindario", value=5000)

    if st.button("Predecir Precio"):
        if modelo:
            try:
                # Crear DataFrame con los datos ingresados
                datos = pd.DataFrame(
                    {
                        "date": [date],
                        "bedrooms": [bedrooms],
                        "bathrooms": [bathrooms],
                        "sqft_living": [sqft_living],
                        "sqft_lot": [sqft_lot],
                        "floors": [floors],
                        "waterfront": [waterfront],
                        "view": [view],
                        "condition": [condition],
                        "grade": [grade],
                        "sqft_above": [sqft_above],
                        "sqft_basement": [sqft_basement],
                        "yr_built": [yr_built],
                        "yr_renovated": [yr_renovated],
                        "lat": [lat],
                        "long": [long],
                        "sqft_living15": [sqft_living15],
                        "sqft_lot15": [sqft_lot15],
                    }
                )

                # Eliminar columnas que no son numéricas (id y date)
                datos = datos.drop(columns=["date"])

                # Asegurar que todas las columnas sean de tipo numérico
                datos = datos.astype(float)

                # Realizar la predicción
                prediccion = modelo.predict(datos)[0]
                st.success(f"El precio estimado de la casa es: ${prediccion:,.2f}")
            except Exception as e:
                st.error(f"Error al procesar los datos: {e}")
        else:
            st.error(
                "El modelo aún no está entrenado. Por favor, actualiza el modelo en la otra pestaña."
            )

with tab2:
    st.header("Subir CSV para Actualizar el Modelo")

    st.markdown(
        """
    - El CSV debe contener todas las columnas necesarias para entrenar el modelo.
    """
    )

    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            # Intentamos leer el archivo CSV con codificación utf-8-sig para manejar correctamente caracteres especiales
            df = pd.read_csv(archivo, delimiter=",", encoding="utf-8-sig")

            # Verificamos si el archivo tiene contenido
            if df.empty:
                st.error("El archivo CSV está vacío.")
            else:
                # Mostrar las primeras filas para ver el contenido
                st.write("Vista previa de los datos cargados:")
                st.dataframe(df.head())

                # Verificar si el archivo contiene las columnas necesarias
                columnas_necesarias = [
                    "date",
                    "bedrooms",
                    "bathrooms",
                    "sqft_living",
                    "sqft_lot",
                    "floors",
                    "waterfront",
                    "view",
                    "condition",
                    "grade",
                    "sqft_above",
                    "sqft_basement",
                    "yr_built",
                    "yr_renovated",
                    "lat",
                    "long",
                    "sqft_living15",
                    "sqft_lot15",
                ]

                if all(col in df.columns for col in columnas_necesarias):
                    # Verificación antes de limpiar los datos
                    st.write(f"Filas originales: {len(df)}")

                    # Limpiar los datos eliminando filas con valores NaN en las columnas críticas
                    df_before_drop = (
                        df.copy()
                    )  # Copia antes de la limpieza para comparación
                    df = df.dropna(subset=columnas_necesarias)
                    st.write(f"Filas después de eliminar filas con NaN: {len(df)}")

                    # Mostrar filas eliminadas
                    filas_eliminadas = df_before_drop[
                        ~df_before_drop.index.isin(df.index)
                    ]
                    if not filas_eliminadas.empty:
                        st.write("Filas eliminadas debido a NaN:")
                        st.dataframe(filas_eliminadas)

                    # Intentar convertir las columnas a tipo numérico (si no puede, se convierte a NaN)
                    for col in columnas_necesarias:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    # Mostrar filas donde se produjo un NaN por conversión a numérico
                    filas_invalidas = df[df[columnas_necesarias].isna().any(axis=1)]
                    if not filas_invalidas.empty:
                        st.write("Filas con datos no numéricos convertidos a NaN:")
                        st.dataframe(filas_invalidas)

                    # Eliminar las filas con NaN generados por la conversión
                    df = df.dropna(subset=columnas_necesarias)
                    st.write(
                        f"Filas después de eliminar filas con NaN por conversión: {len(df)}"
                    )

                    # Limpiar y convertir la columna 'date' a datetime con el formato correcto
                    # Asegurarse de que la columna 'date' es tipo cadena antes de reemplazar 'T'
                    df["date"] = df["date"].astype(
                        str
                    )  # Asegurarse de que 'date' es tipo string
                    df["date"] = df["date"].str.replace("T", "", regex=False)

                    # Ahora, convertir la fecha usando el formato adecuado
                    df["date"] = pd.to_datetime(
                        df["date"], format="%Y%m%d%H%M%S", errors="coerce"
                    )

                    # Mostrar las filas eliminadas debido a fechas no válidas
                    if not df_before_drop[df_before_drop["date"].isna()].empty:
                        st.write("Filas con fechas no válidas eliminadas:")
                        st.dataframe(df_before_drop[df_before_drop["date"].isna()])

                    # Si el DataFrame está vacío después de la limpieza, mostrar un error
                    if df.empty:
                        st.error(
                            "Después de limpiar los datos, no quedan filas válidas para entrenar el modelo."
                        )
                    else:
                        # Separar la columna 'price' como la variable objetivo (y) y las demás como características (X)
                        X = df.drop(columns=["price"])
                        y = df["price"]

                        # Convertir todas las columnas de X a tipo numérico
                        X = X.apply(pd.to_numeric, errors="coerce")

                        if st.button("Entrenar y Guardar Modelo"):
                            st.spinner("Entrenando el modelo...")
                            # Usar el archivo CSV cargado para separar los datos
                            X_train, X_test, y_train, y_test = split_data(archivo)

                            # Guardar el modelo entrenado
                            model_filename = "ml_model_regression.pkl"
                            regressor_new = train_and_save_model(
                                X_train, y_train, model_filename
                            )

                            # Evaluar y comparar el nuevo modelo
                            bucket_name = "datamlops"
                            evaluate_and_compare_models(
                                regressor_new,
                                X_test,
                                y_test,
                                model_filename,
                                bucket_name,
                            )
                            st.success("Modelo entrenado y actualizado correctamente.")
                else:
                    st.error(
                        f"El archivo debe contener las columnas: {', '.join(columnas_necesarias)}"
                    )
        except pd.errors.ParserError as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {e}")

    else:
        st.info("Por favor, sube un archivo CSV para comenzar.")
