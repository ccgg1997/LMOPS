# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# Función para cargar el modelo
def cargar_modelo(modelo_path='../model/ml_model_regression.pkl'):
    if os.path.exists(modelo_path):
        modelo = joblib.load(modelo_path)
        return modelo
    else:
        return None

# Cargar el modelo existente
modelo = cargar_modelo()

st.title("Predicción de Precios de Casas con Machine Learning")

# Crear pestañas
tab1, tab2 = st.tabs(["🔮 Predecir Precio", "📈 Actualizar Modelo"])

with tab1:
    st.header("Ingresar Datos para la Predicción")

    st.markdown("Por favor, ingresa los siguientes parámetros (o usa los valores por defecto):")

    # Parámetros requeridos por el modelo con valores por defecto
    date = st.text_input("Fecha de Venta (YYYY-MM-DD)", value="2023-01-01")
    bedrooms = st.number_input("Número de Habitaciones", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Número de Baños", min_value=1, max_value=10, value=2)
    sqft_living = st.number_input("Metros Cuadrados Habitables", min_value=20, max_value=10000, value=1500)
    sqft_lot = st.number_input("Tamaño del Lote (m²)", min_value=20, max_value=50000, value=5000)
    floors = st.number_input("Número de Pisos", min_value=1, max_value=5, value=1)
    waterfront = st.selectbox("¿Tiene Vista al Agua?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    view = st.number_input("Vista (de 0 a 4)", min_value=0, max_value=4, value=0)
    condition = st.number_input("Condición (de 1 a 5)", min_value=1, max_value=5, value=3)
    grade = st.number_input("Calidad de Construcción (de 1 a 13)", min_value=1, max_value=13, value=7)
    sqft_above = st.number_input("Metros Cuadrados sobre el Nivel del Suelo", min_value=20, max_value=10000, value=1200)
    sqft_basement = st.number_input("Metros Cuadrados en el Sótano", min_value=0, max_value=5000, value=300)
    yr_built = st.number_input("Año de Construcción", min_value=1800, max_value=2025, value=1990)
    yr_renovated = st.number_input("Año de Renovación (0 si no aplica)", min_value=0, max_value=2025, value=0)
    lat = st.number_input("Latitud", value=47.5112)
    long = st.number_input("Longitud", value=-122.257)
    sqft_living15 = st.number_input("Promedio de Metros Habitables en Vecindario", value=1500)
    sqft_lot15 = st.number_input("Promedio de Tamaño de Lote en Vecindario", value=5000)

    if st.button("Predecir Precio"):
        if modelo:
            try:
                # Crear DataFrame con los datos ingresados
                datos = pd.DataFrame({
                    'date': [date],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'sqft_living': [sqft_living],
                    'sqft_lot': [sqft_lot],
                    'floors': [floors],
                    'waterfront': [waterfront],
                    'view': [view],
                    'condition': [condition],
                    'grade': [grade],
                    'sqft_above': [sqft_above],
                    'sqft_basement': [sqft_basement],
                    'yr_built': [yr_built],
                    'yr_renovated': [yr_renovated],
                    'lat': [lat],
                    'long': [long],
                    'sqft_living15': [sqft_living15],
                    'sqft_lot15': [sqft_lot15]
                })

                # Eliminar columnas que no son numéricas (id y date)
                datos = datos.drop(columns=['date'])

                # Asegurar que todas las columnas sean de tipo numérico
                datos = datos.astype(float)

                # Realizar la predicción
                prediccion = modelo.predict(datos)[0]
                st.success(f"El precio estimado de la casa es: ${prediccion:,.2f}")
            except Exception as e:
                st.error(f"Error al procesar los datos: {e}")
        else:
            st.error("El modelo aún no está entrenado. Por favor, actualiza el modelo en la otra pestaña.")

with tab2:
    st.header("Subir CSV para Actualizar el Modelo")

    st.markdown("""
    - El CSV debe contener todas las columnas necesarias para entrenar el modelo.
    """)

    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df = pd.read_csv(archivo)

            # Verificar columnas necesarias
            columnas_necesarias = [
                'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                'sqft_basement', 'yr_built', 'yr_renovated','lat', 
                'long', 'sqft_living15', 'sqft_lot15'
            ]
            if all(col in df.columns for col in columnas_necesarias):
                st.write("Vista previa de los datos cargados:")
                st.dataframe(df.head())

                if st.button("Entrenar y Guardar Modelo"):
                    st.spinner("Entrenando el modelo...")
                    st.info("Función de entrenamiento pendiente de integración.")
            else:
                st.error(f"El archivo debe contener las columnas: {', '.join(columnas_necesarias)}")
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

    else:
        st.info("Por favor, sube un archivo CSV para comenzar.")
