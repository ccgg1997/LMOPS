import os
import sys

# Subir un nivel para acceder al directorio raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.etl import split_data
from pipeline.model_build import train_and_save_model


def main():
    local_file_path = "kc_house_data_qa.csv"
    model_filename = "ml_model_regression_qa.pkl"

    X_train, X_test, y_train, y_test = split_data(local_file_path)
    regressor_qa = train_and_save_model(X_train, y_train, model_filename)

    regressor_qa.fit(X_train, y_train)

    y_pred = regressor_qa.predict(X_test)
    print(f"Predicciones:{y_pred}")


if __name__ == "__main__":
    main()
