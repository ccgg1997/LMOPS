from pipeline.etl import download_and_preprocess_data
from pipeline.model_build import train_and_save_model
from pipeline.model_predict import evaluate_and_compare_models
from datetime import datetime

def main():
    local_file_path = 'data/kc_house_data.csv'
    model_filename = 'ml_model_regression.pkl'
    bucket_name = 'datamlops'

    # Paso 1: Descargar y procesar datos
    X_train, X_test, y_train, y_test = download_and_preprocess_data(local_file_path, bucket_name)

    # Paso 2: Entrenar el modelo y guardar el nuevo modelo
    regressor_new = train_and_save_model(X_train, y_train, model_filename)

    # Paso 3: Evaluar y comparar modelos
    evaluate_and_compare_models(regressor_new, X_test, y_test, model_filename, bucket_name)

if __name__ == '__main__':
    main()
