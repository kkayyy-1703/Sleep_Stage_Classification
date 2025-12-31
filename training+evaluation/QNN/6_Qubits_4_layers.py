# ===================================================
# Full PyTorch + PennyLane implementation of QNN (with early stopping & plots)
# ===================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --------------------------
# PennyLane
# --------------------------
import pennylane as qml
from pennylane import numpy as pnp  # must use Pennylane numpy for QNode tensors

# --------------------------
# Reproducibility
# --------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --------------------------
# Config
# --------------------------
DATA_DIR = r"C:\CP\micro\Dhruv\Motion_HR_Sleep_ClockProxy"  # update path if needed
FEATURES = ['activity_count', 'heartrate_std', 'clock_proxy']
TARGET = 'sleep_stage'
WINDOW_SIZE = 64
LABEL_POS = WINDOW_SIZE // 2
BATCH_SIZE = 64
EPOCHS = 10          # ← changed to 10
NOISE_STD = 0.02

# QNN config
n_qubits = 6
n_q_layers = 4       # ← increased to 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# Load CSVs
# --------------------------
all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
df_list = []
for file in all_files:
    tmp = pd.read_csv(file)
    if set(FEATURES + [TARGET]).issubset(tmp.columns):
        df_list.append(tmp[FEATURES + [TARGET]])
    else:
        print(f"Skipping {file}, missing required cols")

if len(df_list) == 0:
    raise RuntimeError("No CSVs loaded. Check your path!")

df = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(df)} rows from {len(df_list)} files")

# --------------------------
# Clean & scale
# --------------------------
df[FEATURES] = df[FEATURES].interpolate(limit_direction="both")
df = df.dropna(subset=FEATURES + [TARGET])
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

# --------------------------
# Create sequences
# --------------------------
X_seq, y_seq = [], []
N = len(df)
for i in range(N - WINDOW_SIZE):
    win = df.iloc[i:i+WINDOW_SIZE]
    X_seq.append(win[FEATURES].values)
    y_seq.append(int(win[TARGET].iloc[LABEL_POS]))

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print("Sequences created:", X_seq.shape)

# --------------------------
# Encode labels
# --------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y_seq)
class_names = [str(c) for c in le.classes_]
print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# --------------------------
# Oversampling
# --------------------------
X_flat = X_seq.reshape((X_seq.shape[0], -1))
df_bal = pd.DataFrame(X_flat)
df_bal["label"] = y_enc
max_size = df_bal["label"].value_counts().max()
oversampled_list = []
for cls, group in df_bal.groupby("label"):
    oversampled_list.append(resample(group, replace=True, n_samples=max_size, random_state=SEED))
df_balanced = pd.concat(oversampled_list).sample(frac=1, random_state=SEED).reset_index(drop=True)

y_resampled_enc = df_balanced["label"].values
X_resampled = df_balanced.drop(columns="label").values.reshape((-1, WINDOW_SIZE, len(FEATURES)))
y_resampled_cat = np.eye(len(class_names))[y_resampled_enc]

# --------------------------
# Train/test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled_cat, test_size=0.2,
    stratify=y_resampled_enc, random_state=SEED
)

# --------------------------
# Data augmentation (Gaussian noise)
# --------------------------
def add_gaussian_noise(X, std=NOISE_STD):
    return X + np.random.randn(*X.shape) * std

X_train_noisy = add_gaussian_noise(X_train, NOISE_STD)
y_train_noisy = y_train.copy()

X_train_final = np.concatenate([X_train, X_train_noisy], axis=0)
y_train_final = np.concatenate([y_train, y_train_noisy], axis=0)

perm = np.random.RandomState(SEED).permutation(len(X_train_final))
X_train_final = X_train_final[perm]
y_train_final = y_train_final[perm]

# --------------------------
# PyTorch Dataset & Loader
# --------------------------
train_dataset = TensorDataset(torch.tensor(X_train_final, dtype=torch.float32),
                              torch.tensor(y_train_final, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------
# PennyLane QNode
# --------------------------
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_q_layers, n_qubits, 3)}
q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# --------------------------
# Full PyTorch QNN model
# --------------------------
class SleepQNN(nn.Module):
    def __init__(self, n_qubits, n_classes):   # ✅ TWO underscores
        super().__init__()                     # ✅ TWO underscores
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(WINDOW_SIZE * len(FEATURES), 128)
        self.dropout1 = nn.Dropout(0.15)
        self.fc_angles = nn.Linear(128, n_qubits)
        self.q_layer = q_layer
        self.fc2 = nn.Linear(n_qubits, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        angles = self.tanh(self.fc_angles(x)) * torch.pi
        x = self.q_layer(angles)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.out(x)
        return x


model = SleepQNN(n_qubits, len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --------------------------
# Training & Validation Loop (with Early Stopping)
# --------------------------
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_loss = float('inf')
patience = 3
wait = 0
save_path = "best_sleep_qnn_model.pt"

for epoch in range(EPOCHS):
    # ---- Training ----
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, torch.argmax(yb, dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        correct += (torch.argmax(preds, dim=1) == torch.argmax(yb, dim=1)).sum().item()
        total += xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # ---- Validation ----
    model.eval()
    val_running_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_loss = criterion(preds, torch.argmax(yb, dim=1))
            val_running_loss += val_loss.item() * xb.size(0)
            val_correct += (torch.argmax(preds, dim=1) == torch.argmax(yb, dim=1)).sum().item()
            val_total += xb.size(0)
    val_epoch_loss = val_running_loss / len(test_loader.dataset)
    val_epoch_acc = val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")

    # ---- Early Stopping ----
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        wait = 0
        torch.save(model.state_dict(), save_path)
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load(save_path))

# --------------------------
# Plot Training vs Validation Curves
# --------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', linestyle='--')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Loss Curve (Early Stopping)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
plt.plot(range(1, len(val_accs)+1), val_accs, label='Val Acc', linestyle='--')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Accuracy Curve (Early Stopping)')
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# Evaluate on test set
# --------------------------
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        y_true.append(torch.argmax(yb, dim=1).cpu().numpy())
        y_pred.append(torch.argmax(preds, dim=1).cpu().numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

# --------------------------
# Confusion Matrix
# --------------------------
cm_raw = confusion_matrix(y_true, y_pred)
disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=class_names)
disp_raw.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Raw Counts")
plt.tight_layout()
plt.show()

cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
disp_norm.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Normalized (per true class)")
plt.tight_layout()
plt.show()