import streamlit as st
import pandas as pd
import joblib
import os
from model import ml_model_regression

# Función para cargar el modelo
def cargar_modelo(modelo_path='modelo.pkl'):
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

    # Ejemplo de características (ajusta según tu dataset)
    st.markdown("Por favor, ingresa los siguientes parámetros:")

    habitaciones = st.number_input("Número de Habitaciones", min_value=1, max_value=10, value=3)
    banos = st.number_input("Número de Baños", min_value=1, max_value=10, value=2)
    metros_cuadrados = st.number_input("Metros Cuadrados", min_value=20, max_value=1000, value=100)
    edad = st.number_input("Edad de la Casa (años)", min_value=0, max_value=100, value=10)
    ubicacion = st.selectbox("Ubicación", ["Urbana", "Suburbana", "Rural"])

    # Convertir ubicación a variable numérica
    ubicacion_map = {"Urbana": 2, "Suburbana": 1, "Rural": 0}
    ubicacion_num = ubicacion_map[ubicacion]

    if st.button("Predecir Precio"):
        if modelo:
            # Crear DataFrame con los datos ingresados
            datos = pd.DataFrame({
                'habitaciones': [habitaciones],
                'banos': [banos],
                'metros_cuadrados': [metros_cuadrados],
                'edad': [edad],
                'ubicacion': [ubicacion_num]
            })
            prediccion = modelo.predict(datos)[0]
            st.success(f"El precio estimado de la casa es: ${prediccion:,.2f}")
        else:
            st.error("El modelo aún no está entrenado. Por favor, actualiza el modelo en la otra pestaña.")

with tab2:
    st.header("Subir CSV para Actualizar el Modelo")

    st.markdown("""
    - El CSV debe contener las siguientes columnas: `habitaciones`, `banos`, `metros_cuadrados`, `edad`, `ubicacion`, `precio`.
    - La columna `precio` es la variable objetivo.
    - La columna `ubicacion` debe estar codificada como: `Urbana = 2`, `Suburbana = 1`, `Rural = 0`.
    """)

    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df = pd.read_csv(archivo)

            # Verificar que las columnas necesarias estén presentes
            columnas_necesarias = ['habitaciones', 'banos', 'metros_cuadrados', 'edad', 'ubicacion', 'precio']
            if all(col in df.columns for col in columnas_necesarias):
                st.write("Vista previa de los datos cargados:")
                st.dataframe(df.head())

                if st.button("Entrenar y Guardar Modelo"):
                    with st.spinner("Entrenando el modelo..."):
                        mse = entrenar_modelo(
                            df,
                            ['habitaciones', 'banos', 'metros_cuadrados', 'edad', 'ubicacion'],
                            'precio'
                        )
                    st.success(f"Modelo entrenado con un MSE de: {mse}")
                    # Recargar el modelo entrenado
                    modelo = cargar_modelo()
            else:
                st.error(f"El archivo debe contener las columnas: {', '.join(columnas_necesarias)}")
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

    else:
        st.info("Por favor, sube un archivo CSV para comenzar.")
