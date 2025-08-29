# modelo cnn

import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms  # requerido
import lightning as L
import torchmetrics as tm
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve
)

# ========= Dataset =========
class TimeSeriesNPY(Dataset):
    """
    Recibe X: (N, T, F) 
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        assert X.ndim == 3, "X debe ser (N, T, F)"
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.transform = transform

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]              # (T, F)
        x = np.transpose(x, (1, 0))  # -> (F, T) para Conv1d
        x = torch.from_numpy(x)
        if self.transform:
            x = self.transform(x)    # identidad (placeholder)
        y = torch.tensor(self.y[idx])
        return x, y

# ========= Modelo =========
class PyTorchCNN1D(nn.Module):
    """
    CNN-1D ligera para binario:
    Entrada esperada: (B, F, T)
    """
    def __init__(self, n_feats: int, filters: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_feats, filters, kernel_size=3, padding="same")
        self.bn1   = nn.BatchNorm1d(filters)
        self.act1  = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(filters, filters, kernel_size=5, padding="same")
        self.bn2   = nn.BatchNorm1d(filters)
        self.act2  = nn.ReLU()
        self.gap   = nn.AdaptiveMaxPool1d(1)  # Global Max Pool

        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(filters, 64)
        self.act3  = nn.ReLU()
        self.out   = nn.Linear(64, 1)  # logit

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.gap(x).squeeze(-1)          # (B, filters)
        x = self.drop(self.act3(self.fc1(x)))
        logit = self.out(x).squeeze(1)       # (B,)
        return logit

# ========= Lightning Module =========
class LightningCNN1D(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3, pos_weight: float = 1.0):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        # Métricas online
        self.train_acc = tm.classification.BinaryAccuracy()
        self.val_acc   = tm.classification.BinaryAccuracy()
        self.test_acc  = tm.classification.BinaryAccuracy()
        self.val_auc   = tm.classification.AUROC(task="binary")
        self.test_auc  = tm.classification.AUROC(task="binary")

    def forward(self, x): 
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        if stage == "train":
            self.train_acc.update(preds, y.int())
            self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        elif stage == "val":
            self.val_acc.update(preds, y.int())
            self.val_auc.update(probs, y.int())
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        else:
            self.test_acc.update(preds, y.int())
            self.test_auc.update(probs, y.int())
            self.log("test_loss", loss, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):   return self._shared_step(batch, "train")
    def validation_step(self, batch, _): return self._shared_step(batch, "val")
    def test_step(self, batch, _):       return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_auc", self.val_auc.compute(), prog_bar=True)
        self.val_acc.reset(); self.val_auc.reset()

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute())
        self.log("test_auc", self.test_auc.compute())
        self.test_acc.reset(); self.test_auc.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

# ========= Utilidades =========
def compute_pos_weight(y: np.ndarray) -> float:
    """
    pos_weight = (#neg / #pos) para BCEWithLogitsLoss.
    """
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0: 
        return 1.0
    return neg / (pos + 1e-8)

def split_data_with_groups(X, y, groups=None, seed=42):
   
    N = len(y)
    idx_all = np.arange(N)

    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
        tr_idx, tmp_idx = next(gss.split(np.zeros(N), y, groups=groups))
        # Val/Test del 30% restante -> 15/15
        gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
        va_idx, te_idx = next(gss2.split(np.zeros(len(tmp_idx)), y[tmp_idx], groups=groups[tmp_idx]))
        val_idx = tmp_idx[va_idx]; test_idx = tmp_idx[te_idx]
    else:
        # Estratificado por y
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
        tr_idx, tmp_idx = next(sss.split(idx_all, y))
        sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
        va_idx, te_idx = next(sss2.split(tmp_idx, y[tmp_idx]))
        val_idx = tmp_idx[va_idx]; test_idx = tmp_idx[te_idx]

    return tr_idx, val_idx, test_idx

def threshold_max_f1(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    best_i = int(np.argmax(f1))
    # precision_recall_curve devuelve thresholds de tamaño len-1 vs prec/rec
    best_thr = 0.5 if best_i >= len(thr) else thr[best_i]
    return float(best_thr), float(np.max(f1))

# =========
if __name__ == "__main__":
    L.seed_everything(42)

    # 1) Cargar arrays
    X_train = np.load("X_train_cnn.npy")  # (N1, T, F)
    y_train = np.load("y_train_cnn.npy")
    X_test  = np.load("X_test_cnn.npy")   # (N2, T, F)
    y_test  = np.load("y_test_cnn.npy")

   
    groups = None
    if os.path.exists("groups.npy"):
        groups = np.load("groups.npy")
        assert len(groups) == (len(y_train) + len(y_test)), "groups.npy debe tener N_train+N_test elementos."
        # Recombinar para re-split honesto
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        groups_all = groups
    else:
        
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        groups_all = None

  
    tr_idx, val_idx, te_idx = split_data_with_groups(X, y, groups=groups_all, seed=42)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te,  y_te  = X[te_idx], y[te_idx]

    print(f"Split -> train: {len(y_tr)} | val: {len(y_val)} | test: {len(y_te)}")

   
    pos_w = compute_pos_weight(y_tr)
    print("pos_weight (train):", pos_w)

    # DataLoaders (evitar batch de 1)

    dummy_transform = transforms.Lambda(lambda t: t)  # identidad
    train_ds = TimeSeriesNPY(X_tr, y_tr, transform=dummy_transform)
    val_ds   = TimeSeriesNPY(X_val, y_val, transform=dummy_transform)
    test_ds  = TimeSeriesNPY(X_te, y_te, transform=dummy_transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, drop_last=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, drop_last=False, num_workers=0)

    #  Modelo + Lightning
    n_feats = X.shape[2]
    base_model = PyTorchCNN1D(n_feats=n_feats, filters=64, dropout=0.3)
    lightning_model = LightningCNN1D(base_model, lr=1e-3, pos_weight=pos_w)

    # 
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    early = EarlyStopping(monitor="val_auc", mode="max", patience=8)
    ckpt  = ModelCheckpoint(monitor="val_auc", mode="max", save_top_k=1, filename="best-{epoch}-{val_auc:.4f}")
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=80,
        accelerator="auto",
        callbacks=[early, ckpt, lrmon],
        log_every_n_steps=10
    )

    #  Entrenar y cargar 
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_path = ckpt.best_model_path if ckpt.best_model_path else None
    if best_path:
        lightning_model = LightningCNN1D.load_from_checkpoint(best_path, model=base_model, lr=1e-3, pos_weight=pos_w)

    #  Evaluación Lightning rápida
    trainer.test(lightning_model, dataloaders=test_loader)

    #  Inferencia manual para métricas detalladas + umbral óptimo (F1) en VALIDACIÓN
    lightning_model.eval()
    device = next(lightning_model.parameters()).device

    # --- Probabilidades en VALIDACIÓN ---
    y_val_true, y_val_prob = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = lightning_model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            y_val_true.append(yb.numpy().ravel())
            y_val_prob.append(probs)
    y_val_true = np.concatenate(y_val_true).astype(int)
    y_val_prob = np.concatenate(y_val_prob)

    best_thr, best_f1_val = threshold_max_f1(y_val_true, y_val_prob)
    print(f"Umbral óptimo (val, max F1): {best_thr:.4f} | F1_val={best_f1_val:.4f}")

    # --- Probabilidades en TEST ---
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = lightning_model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            y_true_all.append(yb.numpy().ravel())
            y_prob_all.append(probs)

    y_true = np.concatenate(y_true_all).astype(int)
    y_prob = np.concatenate(y_prob_all)

    #  Métricas con threshold óptimo
    y_pred = (y_prob >= best_thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== MÉTRICAS FINALES TEST (umbral valid óptimo) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nMatriz de confusión:\n", cm)
    print("\nReporte de clasificación:\n", classification_report(y_true, y_pred, zero_division=0))
