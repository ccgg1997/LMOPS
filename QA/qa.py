import os
import sys

# Subir un nivel para acceder al directorio raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.etl import split_data
from pipeline.model_build import train_and_save_model


def test_pipeline():
    local_file_path = "kc_house_data_qa.csv"
    model_filename = "ml_model_regression_qa.pkl"

    X_train, X_test, y_train, y_test = split_data(local_file_path)
    regressor_qa = train_and_save_model(X_train, y_train, model_filename)

    regressor_qa.fit(X_train, y_train)

    y_pred = regressor_qa.predict(X_test)
    print(f"Predicciones: {y_pred[0]}")
    assert y_pred is not None and y_pred[0] == float(
        512587.9512686804
    ), "Predicciones incorrectas"

    # Eliminar el archivo del modelo
    if os.path.exists(model_filename):
        os.remove(model_filename)

    print("Prueba completada exitosamente")


if __name__ == "__main__":
    test_pipeline()
