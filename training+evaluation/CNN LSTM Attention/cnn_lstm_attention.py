import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --------------------------
# Load all CSV files
# --------------------------
DATA_DIR = r"/kaggle/input/motion-hr-sleep-clockproxy"
all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

df_list = []
for file in all_files:
    temp = pd.read_csv(file)
    if "sleep_stage" in temp.columns:
        df_list.append(temp)
    else:
        print(f"⚠ Skipping {file}, missing 'sleep_stage' column")

df = pd.concat(df_list, ignore_index=True)
print(f"✅ Loaded {len(df)} rows from {len(df_list)} files")

# --------------------------
# Preprocessing
# --------------------------
df = df.interpolate(limit_direction="both")
df = df.dropna(subset=["sleep_stage"])

target = "sleep_stage"
features = [c for c in df.columns if c != target and df[c].dtype != "object"]
print("Using features:", features)

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# --------------------------
# Make sequences
# --------------------------
window_size = 20
X_seq, y_seq = [], []

for i in range(len(df) - window_size):
    X_seq.append(df[features].iloc[i:i+window_size].values)
    y_seq.append(df[target].iloc[i+window_size-1])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# --------------------------
# Label encode
# --------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y_seq)
class_names = [str(c) for c in le.classes_]
print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# --------------------------
# Oversample
# --------------------------
X_seq_flat = X_seq.reshape((X_seq.shape[0], -1))
data = pd.DataFrame(X_seq_flat)
data["label_enc"] = y_enc

max_size = data["label_enc"].value_counts().max()
lst = [data]

for cls, group in data.groupby("label_enc"):
    if len(group) < max_size:
        lst.append(group.sample(max_size - len(group), replace=True, random_state=42))

data_balanced = pd.concat(lst, ignore_index=True)
data_balanced = shuffle(data_balanced, random_state=42)

y_resampled_enc = data_balanced["label_enc"].values
X_resampled = data_balanced.drop(columns="label_enc").values
X_resampled = X_resampled.reshape((-1, window_size, len(features)))
y_resampled_cat = to_categorical(y_resampled_enc)

# --------------------------
# Train-test split
# --------------------------
X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
    X_resampled, y_resampled_cat, y_resampled_enc,
    test_size=0.2, stratify=y_resampled_enc, random_state=42
)

# --------------------------
# Build Deep CNN + LSTM
# --------------------------
model = Sequential([
    Conv1D(64, 3, activation="relu", input_shape=(window_size, X_train.shape[2])),
    Conv1D(128, 3, activation="relu"),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(256, 3, activation="relu"),
    MaxPooling1D(2),
    Dropout(0.3),

    LSTM(128),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dense(y_train.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5)
]

# --------------------------
# Train model
# --------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# --------------------------
# Evaluate
# --------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

# Predictions
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_enc = np.argmax(y_pred_probs, axis=1)
y_true_enc = np.argmax(y_test, axis=1)

# --------------------------
# Classification Report
# --------------------------
print("\nClassification Report:")
print(classification_report(
    y_true_enc, y_pred_enc,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    digits=4
))

# --------------------------
# Confusion Matrices
# --------------------------
cm = confusion_matrix(y_true_enc, y_pred_enc)
cm_norm = confusion_matrix(y_true_enc, y_pred_enc, normalize="true")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    cmap="Blues", xticks_rotation=45, ax=plt.gca()
)
plt.title("Confusion Matrix")

plt.subplot(1,2,2)
ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names).plot(
    cmap="Blues", xticks_rotation=45, ax=plt.gca()
)
plt.title("Normalized Confusion Matrix")

plt.tight_layout()
plt.show()

# --------------------------
# Accuracy & Loss Curves
# --------------------------
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Val"])

# Loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Val"])

plt.tight_layout()
plt.show()

# --------------------------
# AUC (Macro)
# --------------------------
try:
    auc_macro = roc_auc_score(y_test, y_pred_probs, multi_class='ovo', average="macro")
    print(f"\nMacro AUC: {auc_macro:.4f}")
except:
    print("AUC could not be computed.")