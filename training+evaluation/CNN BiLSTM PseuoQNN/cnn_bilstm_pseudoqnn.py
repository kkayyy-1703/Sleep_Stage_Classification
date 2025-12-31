import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# reproducibility
import random, tensorflow as tf
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.utils import shuffle, resample, class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, Layer, Concatenate
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

# --------------------------
# Config
# --------------------------
DATA_DIR = r"/kaggle/input/sleep-motion-hr"   # <-- Update path if needed
FEATURES = ['activity_count', 'heartrate_std']
TARGET = 'sleep_stage'

WINDOW_SIZE = 64            # mid window labeling
LABEL_POS = WINDOW_SIZE // 2
L2_REG = 1e-4
BATCH_SIZE = 64
EPOCHS = 50
NOISE_STD = 0.02           # augmentation noise level

# --------------------------
# Load all CSV files
# --------------------------
all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
df_list = []
for file in all_files:
    tmp = pd.read_csv(file)
    if set(FEATURES + [TARGET]).issubset(tmp.columns):
        df_list.append(tmp[FEATURES + [TARGET]])
    else:
        print(f"⚠ Skipping {os.path.basename(file)}, missing required cols")

if len(df_list) == 0:
    raise RuntimeError(f"No valid CSV files found in {DATA_DIR} with required columns {FEATURES + [TARGET]}")

df = pd.concat(df_list, ignore_index=True)
print(f"✅ Loaded {len(df)} rows from {len(df_list)} files")

# --------------------------
# Clean / scale
# --------------------------
df[FEATURES] = df[FEATURES].interpolate(limit_direction="both")
df = df.dropna(subset=FEATURES + [TARGET])

scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

# --------------------------
# Create sequences (label = middle timestep)
# --------------------------
X_seq, y_seq = [], []
N = len(df)
for i in range(0, N - WINDOW_SIZE):
    win = df.iloc[i:i+WINDOW_SIZE]
    X_seq.append(win[FEATURES].values)
    y_seq.append(int(win[TARGET].iloc[LABEL_POS]))

X_seq = np.array(X_seq)   # (samples, WINDOW_SIZE, n_features)
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
# Manual oversampling (no imblearn) - balance classes
# --------------------------
X_flat = X_seq.reshape((X_seq.shape[0], -1))
df_bal = pd.DataFrame(X_flat)
df_bal["label"] = y_enc

max_size = df_bal["label"].value_counts().max()
oversampled_list = []
for cls, group in df_bal.groupby("label"):
    oversampled_list.append(resample(group,
                                     replace=True,
                                     n_samples=max_size,
                                     random_state=SEED))
df_balanced = pd.concat(oversampled_list).sample(frac=1, random_state=SEED).reset_index(drop=True)

y_resampled_enc = df_balanced["label"].values
X_resampled = df_balanced.drop(columns="label").values.reshape((-1, WINDOW_SIZE, len(FEATURES)))
y_resampled_cat = to_categorical(y_resampled_enc)

print("After oversampling class counts:", np.bincount(y_resampled_enc))

# --------------------------
# Train/test split
# --------------------------
X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
    X_resampled, y_resampled_cat, y_resampled_enc,
    test_size=0.2, stratify=y_resampled_enc, random_state=SEED
)

# --------------------------
# Class weights (computed from training encoded labels)
# --------------------------
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
class_weights = {i: w for i, w in enumerate(cw)}
print("Class weights:", class_weights)

# --------------------------
# Data augmentation (train only) - Gaussian noise
# --------------------------
def add_gaussian_noise(X, std=NOISE_STD):
    return X + np.random.randn(*X.shape) * std

X_train_noisy = add_gaussian_noise(X_train, std=NOISE_STD)
y_train_noisy = y_train.copy()

# Concatenate original + augmented train data
X_train_final = np.concatenate([X_train, X_train_noisy], axis=0)
y_train_final = np.concatenate([y_train, y_train_noisy], axis=0)

# Shuffle training data
perm = np.random.RandomState(SEED).permutation(len(X_train_final))
X_train_final = X_train_final[perm]
y_train_final = y_train_final[perm]

print("Training samples after augmentation:", X_train_final.shape[0])

# --------------------------
# Self-Attention Layer (fixed __init__)
# --------------------------
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True, name="W_att")
        self.v = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True, name="v_att")
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        u = K.tanh(K.dot(inputs, self.W))                  # (batch, timesteps, features)
        scores = K.dot(u, self.v)                          # (batch, timesteps, 1)
        scores = K.squeeze(scores, -1)                     # (batch, timesteps)
        alpha = K.softmax(scores, axis=-1)                 # (batch, timesteps)
        attended = inputs * K.expand_dims(alpha, axis=-1) # (batch, timesteps, features)
        return K.sum(attended, axis=1)                     # (batch, features)

    def get_config(self):
        base_cfg = super(SelfAttention, self).get_config()
        return base_cfg

# --------------------------
# Pseudo-QNN Layer (fixed __init__)
# --------------------------
class PseudoQNN(Layer):
    def __init__(self, units, **kwargs):
        super(PseudoQNN, self).__init__(**kwargs)
        self.units = int(units)

    def build(self, input_shape):
        self.theta = self.add_weight(
            name="theta", shape=(input_shape[-1], self.units),
            initializer="glorot_uniform", trainable=True
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.units,),
            initializer="zeros", trainable=True
        )
        super(PseudoQNN, self).build(input_shape)

    def call(self, inputs):
        x = K.dot(inputs, self.theta) + self.bias
        return K.sin(x)

    def get_config(self):
        cfg = super(PseudoQNN, self).get_config()
        cfg.update({"units": self.units})
        return cfg

# --------------------------
# Build model: CNN -> BiLSTM -> SelfAttention -> PseudoQNN -> Output
# --------------------------
inp = Input(shape=(WINDOW_SIZE, len(FEATURES)))

x = Conv1D(128, 5, activation='relu', padding='same',
           kernel_regularizer=regularizers.l2(L2_REG))(inp)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Conv1D(256, 5, activation='relu', padding='same',
           kernel_regularizer=regularizers.l2(L2_REG))(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.35)(x)

x = Bidirectional(LSTM(128, return_sequences=True,
                       kernel_regularizer=regularizers.l2(L2_REG)))(x)
x = Dropout(0.35)(x)
x = Bidirectional(LSTM(64, return_sequences=True,
                       kernel_regularizer=regularizers.l2(L2_REG)))(x)
x = Dropout(0.25)(x)

att = SelfAttention()(x)
x = GlobalAveragePooling1D()(x)
x = Concatenate()([x, att])

x = PseudoQNN(64)(x)
x = Dropout(0.4)(x)

out = Dense(y_train.shape[1], activation='softmax')(x)
model = Model(inputs=inp, outputs=out)

# --------------------------
# Optimizer
# --------------------------
try:
    from tensorflow.keras.optimizers import AdamW
    optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-5)
    print("Using AdamW optimizer")
except Exception:
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=1e-3)
    print("AdamW unavailable — falling back to Adam")

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --------------------------
# Callbacks
# --------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

# --------------------------
# Train
# --------------------------
history = model.fit(
    X_train_final, y_train_final,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# --------------------------
# Evaluate
# --------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc:.4f}")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred_enc = np.argmax(y_pred_probs, axis=1)
y_true_enc = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_enc, y_pred_enc, labels=np.arange(len(class_names)),
                            target_names=class_names, digits=4, zero_division=0))

macro_f1 = f1_score(y_true_enc, y_pred_enc, average='macro')
weighted_f1 = f1_score(y_true_enc, y_pred_enc, average='weighted')
print(f"Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

# --------------------------
# Confusion Matrices
# --------------------------
cm_raw = confusion_matrix(y_true_enc, y_pred_enc)
disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=le.classes_)
disp_raw.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Raw Counts")
plt.tight_layout()
plt.show()

cm_norm = confusion_matrix(y_true_enc, y_pred_enc, normalize='true')
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=le.classes_)
disp_norm.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Normalized (per true class)")
plt.tight_layout()
plt.show()

# --------------------------
# Training curves
# --------------------------
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curve')
plt.tight_layout(); plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.tight_layout()
plt.show()