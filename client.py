import argparse
import warnings
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)

from task import load_and_preprocess_dataset, save_log_client, create_cn_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# Define argumentos para el parser: id del cliente
parser = argparse.ArgumentParser()
parser.add_argument(
    "--partition-id",
    default=0,
    type=int,
    help="Partition ID used for the current client.",
)
# Define el path al dataset del cliente
parser.add_argument(
    "--dataset-path",
    type=str,
    required=True,
    help="Path to the dataset file.",
)
args = parser.parse_args()

# Hiperparametros iniciales para el modelo XGBoost
num_local_round = 20
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

# Almacena el path ingresado por consola
dataset_path = args.dataset_path
# Almacena el id cliente ingresado por consola
partition_id = args.partition_id

# Carga y convierte a formato matriz el dataset del cliente
train_dmatrix, valid_dmatrix, num_train, num_val = load_and_preprocess_dataset(dataset_path)

# Cliente XGBoost
class XgbClient(fl.client.Client):
    def __init__(self):
        self.bst = None
        self.config = None

    # Obtiene los parametros del modelo XGBoost
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    # Proceso actualización de parametros modelo local
    def _local_boost(self):
        # Actualiza el modelo con los datos locales
        for i in range(num_local_round):
            self.bst.update(train_dmatrix, self.bst.num_boosted_rounds())

        # Copia del modelo local para agregación o evaluación
        bst = self.bst[
              self.bst.num_boosted_rounds()
              - num_local_round: self.bst.num_boosted_rounds()
              ]

        return bst

    # Inicia proceso de entrenamientos
    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # Entra el modelo XGBoost con los datos de entrenamiento y envia dataset de evaluación
            bst = xgb.train(
                # parametros iniciales
                params,
                # Matriz con datos de entranamiento
                train_dmatrix,
                # Número de rondas locales
                num_boost_round=num_local_round,
                # Configura matriz de validacion para calculo de métrica por defecto
                evals=[(valid_dmatrix, "validate"), (train_dmatrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Carga el modelo global enviado por el servidor y lo configura
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)
            # Inicia actualización del modelo
            bst = self._local_boost()

        # Guarda modelo actualizado
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        # Regresa el modelo actualizado al servidor en formato bytes
        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=num_train,
            metrics={},
        )
    # Función para evaluar el modelo localmente
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Obtiene métricas definidas en la configuración de parametros
        eval_results = self.bst.eval_set(
            evals=[(valid_dmatrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        # Almacena el número de ronda
        round_number = ins.config.get("round", -1)

        # Almacena la métrica AUC sobre el conjunto de evaluación
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        # Realizar predicciones para calculo de métricas adicionales
        y_pred_prob = self.bst.predict(valid_dmatrix)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        y_true = valid_dmatrix.get_label()

        # Calcula métricas adicionales para estudio posterior
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_pred_prob),
        }
        # Calcula el rendimiento por cada clase de la variable objetivo
        print(classification_report(y_true, y_pred))

        # Guarda los registros de métricas del cliente por cada ronda
        save_log_client(round_number, metrics, partition_id)
        # Si es la última ronda calcula la matriz de confusión con la última copia del modelo
        if round_number == 100:
            create_cn_matrix(round_number, y_true, y_pred, "resultados_fl/cliente_"+str(partition_id))


        # Regresa estado y métrica AUC por defecto al servidor
        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=num_val,
            metrics={"AUC": auc},
        )

# Inicia el cliente XGBoost
fl.client.start_client(server_address="127.0.0.1:8085", client=XgbClient().to_client())