
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # requerido
import lightning as L
from torch import nn, optim
import torchmetrics as tm


class TimeSeriesNPY(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        assert X.ndim == 3, "X debe ser (N, T, F)"
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.transform = transform

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]              
        x = np.transpose(x, (1, 0))  
        x = torch.from_numpy(x)
        if self.transform:
            x = self.transform(x)    
        y = torch.tensor(self.y[idx])
        return x, y

# Carga de arrays
X_train = np.load("X_train_cnn.npy")   
y_train = np.load("y_train_cnn.npy")
X_test  = np.load("X_test_cnn.npy")
y_test  = np.load("y_test_cnn.npy")

dummy_transform = transforms.Lambda(lambda t: t)
train_ds = TimeSeriesNPY(X_train, y_train, transform=dummy_transform)


from torch.utils.data import random_split
L.seed_everything(42)
n_total = len(train_ds)
n_val = max(1, int(0.2 * n_total))
n_train = n_total - n_val
train_split, val_split = random_split(train_ds, lengths=[n_train, n_val])

test_ds = TimeSeriesNPY(X_test, y_test, transform=dummy_transform)

train_loader = DataLoader(train_split, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_split,   batch_size=64, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,     batch_size=64, shuffle=False, num_workers=0)

X_train.shape


# === Modelo
class PyTorchCNN1D(nn.Module):
    def __init__(self, n_feats: int, filters: int = 64, dropout: float = 0.3):
        super().__init__()
        # Entrada: (B, F, T)
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
        self.out   = nn.Linear(64, 1)  # logit binario

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.gap(x).squeeze(-1)   # (B, filters)
        x = self.drop(self.act3(self.fc1(x)))
        logit = self.out(x).squeeze(1)  # (B,)
        return logit


class LightningCNN1D(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = tm.classification.BinaryAccuracy()
        self.val_acc   = tm.classification.BinaryAccuracy()
        self.test_acc  = tm.classification.BinaryAccuracy()
        self.val_auc   = tm.classification.AUROC(task="binary")
        self.test_auc  = tm.classification.AUROC(task="binary")

    def forward(self, x): return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        if stage == "train":
            self.train_acc.update(preds, y.int())
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        elif stage == "val":
            self.val_acc.update(preds, y.int())
            self.val_auc.update(probs, y.int())
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.test_acc.update(preds, y.int())
            self.test_auc.update(probs, y.int())
            self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, _):   return self._shared_step(batch, "train")
    def validation_step(self, batch, _): return self._shared_step(batch, "val")
    def test_step(self, batch, _):       return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log("val_acc",  self.val_acc.compute(),  prog_bar=True)
        self.log("val_auc",  self.val_auc.compute(),  prog_bar=True)
        self.val_acc.reset(); self.val_auc.reset()

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute())
        self.log("test_auc", self.test_auc.compute())
        self.test_acc.reset(); self.test_auc.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

# ===  Entrenamiento y test 
n_feats = X_train.shape[2]
base_model = PyTorchCNN1D(n_feats=n_feats, filters=64, dropout=0.3)
lightning_model = LightningCNN1D(base_model, lr=1e-3)

trainer = L.Trainer(
    max_epochs=40,
    accelerator="auto",
    log_every_n_steps=10
)
trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(lightning_model, dataloaders=test_loader)

### metricas

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)

lightning_model.eval()
device = next(lightning_model.parameters()).device

y_true_all, y_prob_all = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = lightning_model(xb)           # (B,)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        y_true_all.append(yb.numpy().ravel())
        y_prob_all.append(probs)

y_true = np.concatenate(y_true_all).astype(int)
y_prob = np.concatenate(y_prob_all)
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)
auc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)

print("\n=== MÉTRICAS FINALES TEST ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nMatriz de confusión:\n", cm)
print("\nReporte de clasificación:\n", classification_report(y_true, y_pred, zero_division=0))


