
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score



input_columns = ['From Bank','To Bank','Amount Received','Amount Paid']

data = pd.read_csv('datasets/data3.csv')

# Define input features (X) and target variable (y)
X_train = data[input_columns]
y_train = data['Is Laundering']

data_test = pd.read_csv('datasets/data4.csv')
# Define input features (X) and target variable (y)
X_test = data_test[input_columns]
y_test = data_test['Is Laundering']


# Modelo XGBoost para clasificación binaria
model = XGBClassifier(
    max_depth=11,         # Profundidad máxima de los árboles
    learning_rate=0.1564960673045264,   # Tasa de aprendizaje
    subsample=0.6690385782792252,       # Porcentaje de datos para cada árbol
    colsample_bytree=0.9564455562076204, # Porcentaje de características para cada árbol
    objective='binary:logistic', # Clasificación binaria
    gamma=0.00587249906130049,
    min_child_weight = 9,
    reg_lambda = 0.012542997268376503,
    alpha = 2.837834277207051,
    eval_metric='auc',   # Métrica de evaluación
    random_state=42      # Semilla para reproducibilidad
)

model.fit(X_train,y_train)

#Predictions
y_pred = model.predict(X_test)

# Calcula matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Crea visualización de la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap="YlOrRd", values_format="d")

# Configura y guardar la imagen
plt.title(f"Matriz de Confusión - Modelo Centralizado 3")
plt.savefig("resultados_centralizado/matriz_centralizada.png")
plt.close()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Calcular probabilidades predichas
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Extraer la columna de la clase positiva
y_pred2 = (y_pred_proba >= 0.5).astype(int)

metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_score": roc_auc_score(y_test, y_pred_proba)
    }

# Guarda las métricas en un archivo CSV
pd.DataFrame([metrics]).to_csv("resultados_centralizado/metricas.csv", index=False)


