import pickle

from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_and_save_model(X_train, y_train, model_filename):
    model_path = f"model/{model_filename}"
    regressor = train_model(X_train, y_train)
    with open(model_path, "wb") as file:
        pickle.dump(regressor, file)
    print(f"Modelo guardado en {model_path}.")
    return regressor
