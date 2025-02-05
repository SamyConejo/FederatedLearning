import flwr as fl
import pandas as pd
from flwr.server.strategy import FedXgbBagging
from flwr.common import Parameters, Scalar
import numpy as np
import time
import xgboost as xgb
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report

from task import save_log_server, create_cn_matrix


# Configuración del participantes del esquema federado
pool_size = 3
num_rounds = 100
num_clients_per_round = 3
num_evaluate_clients = 3

# Configuración inicial del modelo
params = {
    'objective': 'binary:logistic',           # Clasificación binaria
    'max_depth': 11,                          # Profundidad máxima del árbol
    'eta': 0.1564960673045264,                # Tasa de aprendizaje
    'subsample': 0.6690385782792252,          # Fracción de datos para cada árbol
    'colsample_bytree': 0.9564455562076204,   # Fracción de características por árbol
    'gamma': 0.00587249906130049,             # Pérdida mínima para dividir nodos
    'min_child_weight': 9,                    # Peso mínimo de nodos hijos
    'reg_lambda': 0.012542997268376503,       # Regularización L2
    'reg_alpha': 2.837834277207051,           # Regularización L1
    'eval_metric': 'auc',                     # Métrica de evaluación
    'seed': 42,                               # Semilla para reproducibilidad
}

x_auc=np.arange(1,num_rounds+1)
y_auc=[]
time_taken=[]



# Carga dataset de prueba para evaluar el modelo centralizado en servidor
data = pd.read_csv('datasets/data4.csv', sep=";|,", engine='python')

# Define columnas de interes
input_columns = ['From Bank','To Bank','Amount Received','Amount Paid']
# Escoge atributos de entrada
X = data[input_columns].values
# Define atributo de salida
y = data['Is Laundering'].values

# Convierte el dataset en matriz para manejar XGBoost
test_data = xgb.DMatrix(X, label=y, enable_categorical=True)

# Función para calcula metricas agregadas recibidas desde clientes
def evaluate_metrics_aggregation(eval_metrics):
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}
    y.append(auc_aggregated)
    print("\nAUC Score : ",auc_aggregated)
    print("\n")
    return metrics_aggregated

# Funcion customizada para evaluar modelo centralizado en servidor con métricas adicionales
def get_evaluate_fn(test_data, params):

    # Regresa métrica calculada
    def evaluate_fn(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        # Si es la primera ronda regresa 0
        if server_round == 0:
            return 0, {}
        else:
            # Inicia el modelo en servidor con parametros iniciales
            bst = xgb.Booster(params=params)

            # Recupera parametros actualizados con datos recibidos de clientes
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Carga el modelo centralizada con parametros federados
            bst.load_model(para_b)

            # Evalua el modelos con métrica AUC por defecto
            eval_results = bst.eval_set(
                evals=[(test_data, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
            # Formatea métrica AUC
            auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

            #
            y_auc.append(auc)
            # Extraer el AUC calculado por XGBoost
            auc_xgb = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

            # Realizar predicciones
            y_pred_prob = bst.predict(test_data)  # Probabilidades

            print("ACUMULACION:", test_data.num_row())
            y_pred = (y_pred_prob >= 0.5).astype(int)  # Etiquetas binarias

            # Cargar las etiquetas reales
            y_true = test_data.get_label()

            # Calcula métricas adicionales para estudio posterior
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "auc": roc_auc_score(y_true, y_pred_prob),
            }

            # Guarda los registros de métricas del servidor por cada ronda
            save_log_server(server_round, metrics)

            # Generar y guardar matriz de confusión si es la ronda 20
            if server_round == num_rounds:
                create_cn_matrix(server_round, y_true, y_pred, "resultados_fl/server")

            print(classification_report(y_true, y_pred))

            return 0, {"AUC": auc}

    return evaluate_fn

# Función para incluir la configuración del número de ronda en fit
def fit_config(server_round: int):
    return {"round": server_round}

# Función para incluir la configuración del número de ronda en evaluate
def evaluate_config(server_round: int):
    return {"round": server_round}

# Define estrategia de agregación federada
strategy = FedXgbBagging(
    evaluate_function=get_evaluate_fn(test_data,params),
    fraction_fit=(float(num_clients_per_round) / pool_size),
    min_fit_clients=num_clients_per_round,
    min_available_clients=pool_size,
    min_evaluate_clients=num_evaluate_clients,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=None,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,  #
)
# Inicia conteo de tiempo de aprendizaje federado
start_time = time.time()

# Inicia el servidor Flower con la estrategia definida
fl.server.start_server(
    server_address="0.0.0.0:8085",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)

# Muestra y almacena tiempo de aprendizaje federado
time_taken.append(time.time() - start_time)
print("Time taken:", time_taken[0], "seconds")

