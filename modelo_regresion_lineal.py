import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# ========= Cargar datos =========
PATH_DATA = "caracteristica_telco.csv"   
df = pd.read_csv(PATH_DATA)

# ======= Analisis vector de caracteristicas =========
print("\n== Dimensiones y columnas ==")
print(df.shape)
print(df.columns.tolist())

print("\n== Balance de la etiqueta (y_international) ==")
print(df["y_international"].value_counts())

print("\n== Tipos de datos ==")
print(df.dtypes)

# == Preparar X, y (solo numéricas)

target = "y_international"
cols_a_excluir = {"y_international", "y_country", "countrycode", "datetime"}  # datetime es string

# Tomamos solo columnas numéricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Quitamos targets y columnas a excluir
X_cols = [c for c in num_cols if c not in cols_a_excluir]
X = df[X_cols].copy()
y = df[target].astype(int).values

print("\n== Nº de característica usadas ==")
print(len(X_cols))
print("Features:", X_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, stratify=y, random_state=42
)

# ===== ajustar train
scaler = StandardScaler()
scaler.fit(X_train)                # ajusta en entrenamiento
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ========= 6) CV MANUAL p
Cs = [0.01, 0.1, 1, 10]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_score_for_C(C_value):
    f1s = []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        # 
        sc = StandardScaler()
        sc.fit(X_tr)
        X_tr_s = sc.transform(X_tr)
        X_va_s = sc.transform(X_va)

        clf = LogisticRegression(
            C=C_value, penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=1000
        )
        clf.fit(X_tr_s, y_tr)
        y_va_pred = clf.predict(X_va_s)

        _, _, f1, _ = precision_recall_fscore_support(
            y_va, y_va_pred, average="binary", zero_division=0
        )
        f1s.append(f1)
    return np.mean(f1s), np.std(f1s)

cv_results = []
for C in Cs:
    m, s = cv_score_for_C(C)
    cv_results.append((C, m, s))
    print(f"C={C}: F1 mean={m:.4f} ± {s:.4f}")

best_C, best_mean, best_std = max(cv_results, key=lambda t: t[1])
print(f"\n== Mejor C (según F1-CV) => C={best_C} (F1={best_mean:.4f}) ==")

# === Entrenamiento FINAL  (ajustado con todo el train) =========
clf = LogisticRegression(
    C=best_C, penalty="l2", solver="lbfgs",
    class_weight="balanced", max_iter=1000
)
clf.fit(X_train_s, y_train)

# Evaluación =========

y_pred = clf.predict(X_test_s)
if hasattr(clf, "predict_proba"):
    y_prob = clf.predict_proba(X_test_s)[:, 1]
else:
    # En caso de no tener predict_proba, usar decision_function
    from sklearn.utils.extmath import softmax
    scores = clf.decision_function(X_test_s)
    # Aproximación 
    y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n== Métricas en TEST ==")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\n== Reporte de clasificación ==")
print(classification_report(y_test, y_pred, zero_division=0))

print("== Matriz de confusión ==")
print(confusion_matrix(y_test, y_pred))

# ===== Importancia (coeficientes) =========
coefs = pd.Series(clf.coef_.ravel(), index=X_cols).sort_values(ascending=False)
print("\n== Top +coef (empujan a clase 1: internacional) ==")
print(coefs.head(10))
print("\n== Top -coef (empujan a clase 0: nacional) ==")
print(coefs.tail(10))

#  Curva ROC
fpr, tpr, thr = roc_curve(y_test, y_prob)
roc_points = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
print("\n== Primeros puntos de la curva ROC ==")
print(roc_points.head())