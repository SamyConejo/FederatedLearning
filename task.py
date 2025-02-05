import os
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def save_log_client(ronda, metrics, client_id):
    # Crea la ruta del subdirectorio del cliente
    subdirectorio_cliente = f"resultados_fl/cliente_{client_id}"
    os.makedirs(subdirectorio_cliente, exist_ok=True)  # Crear el subdirectorio si no existe

    # Crea la ruta completa del archivo de métricas
    filename = os.path.join(subdirectorio_cliente, f"metrics_ronda_{ronda}.csv")

    # Guarda las métricas en un archivo CSV
    pd.DataFrame([metrics]).to_csv(filename, index=False)

    print(f"Métricas guardadas en: {filename}")


def save_log_server(ronda, metrics):
    # Crea la ruta del subdirectorio del cliente
    subdirectorio_cliente = f"resultados_fl/server"
    os.makedirs(subdirectorio_cliente, exist_ok=True)  # Crear el subdirectorio si no existe

    # Crea la ruta completa del archivo de métricas
    filename = os.path.join(subdirectorio_cliente, f"metrics_ronda_{ronda}.csv")

    # Guarda las métricas en un archivo CSV
    pd.DataFrame([metrics]).to_csv(filename, index=False)

    print(f"Métricas guardadas en: {filename}")


def create_cn_matrix(server_round, y_true, y_pred, output_path):
    # Calcula matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    # Crea visualización de la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")

    # Configura y guardar la imagen
    plt.title(f"Matriz de Confusión - Ronda {server_round}")
    output_file = os.path.join(output_path, f"confusion_matrix_ronda_{server_round}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Matriz de confusión guardada en: {output_file}")


# Función para cargar datasets y transformarlos en DMatrix para XGBoost
def load_and_preprocess_dataset(dataset_path):
    # Carga el dataset
    data = pd.read_csv(dataset_path, sep=";|,", engine='python')

    # Define los atributos de interes
    input_columns = ['From Bank', 'To Bank', 'Amount Received', 'Amount Paid']

    # Define atributos de entrada
    X = data[input_columns].values
    # Define variable objetivo
    y = data['Is Laundering'].values

    # Cree el conjunto de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transforma las muestras a una matriz DMatrix para XGBoost
    train_dmatrix = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    valid_dmatrix = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    # Almacena el numero de registros de cada conjunto
    num_train = len(X_train)
    num_val = len(X_test)

    # Regresa conjuntos de entramiento y validación
    return train_dmatrix, valid_dmatrix, num_train, num_val
