import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# === 1. Cargar dataset con features (punto C) ===
PATH_DATA = "caracteristica_telco.csv"   # <-- ajusta a tu ruta real
df = pd.read_csv(PATH_DATA)

# Objetivo binario
y = df["y_international"]

# Quitamos columnas que no sirven como predictores
cols_to_drop = ["y_country", "y_international", "countrycode", "datetime"]
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# === 2. División de datos ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 3. Pipeline: Escalado + Regresión Logística ===
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# === 4. Grid de hiperparámetros ===
param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "saga"],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# === 5. Entrenamiento ===
grid.fit(X_train, y_train)

# === 6. Evaluación ===
y_pred = grid.predict(X_test)
y_prob = grid.predict_proba(X_test)[:,1]

print("== Mejor modelo encontrado ==")
print(grid.best_params_)

print("\n== Reporte de clasificación ==")
print(classification_report(y_test, y_pred))

print("\n== Matriz de confusión ==")
print(confusion_matrix(y_test, y_pred))

print("\n== ROC-AUC ==")
print(roc_auc_score(y_test, y_prob))
