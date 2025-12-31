# Sleep Stage Classification — Project README

This repository contains code, data pointers, and reproduction instructions for wearable-based sleep stage classification experiments. The project implements and extends methods from two accompanying papers: a CNN–LSTM baseline and a CNN–BiLSTM–Self-Attention–PseudoQNN hybrid, and a separate study exploring Quantum Neural Network (QNN) variants. The codebase contains preprocessing, training, evaluation, and optional quantum-hybrid experiments.

> **Novelty note (careful):** To our knowledge, QNN-style and pseudo-quantum hybrid layers have not been widely explored for wearable-derived sleep-stage classification. These experiments may be among the earliest investigations of QNNs and PseudoQNNs in this specific application area; the claim is offered cautiously and should be validated against future literature.

---

## Repository layout

```
Sleep-Stage-Classification/
│
├── data/
│   ├── raw/                # Raw CSV exports (wearable streams) — source repo noted below
│   └── processed/          # Preprocessed numpy / csv files
│
├── models/                 # placeholder: trained models (not included)
│   └── README.md
│
├── training.py             # baseline training pipeline (or cnn_lstm_attention.py)
├── cnn_lstm_attention.py   # baseline + attention (if present)
├── cnn_bilstm_pseudoqnn.py # advanced hybrid model training script
├── train_qnn.py            # QNN / quantum-hybrid experiments (or 6_Qubits_4_layers.py)
├── evaluation.py           # plotting & metrics
│
├── qnn/                    # (optional) PennyLane / QNN utilities
│
├── results/                # accuracy/loss plots, confusion matrices
│
└── README.md               # this file
```

---

## One-line summary

Train and evaluate deep and hybrid models (CNN–LSTM baseline, CNN–BiLSTM–Self-Attention–PseudoQNN hybrid, and quantum-inspired QNN variants) on wearable-derived signals (activity counts, heart-rate features, and a clock proxy) to predict multiple sleep stages (Wake, N1, N2, N3, REM).

---

## Data source

* The raw dataset used for preprocessing in this repository was taken from: [https://github.com/ojwalch/sleep_accel](https://github.com/ojwalch/sleep_accel)
* Place the downloaded CSVs (from the source repository or your local exports) into `data/raw/`. Preprocessed and ready-to-train files should be placed in `data/processed/` if you prefer to skip the preprocessing step.
* The primary modalities in the raw files are accelerometer (or derived activity counts), heart-rate/PPG features, steps (used to form a clock proxy), and PSG-derived sleep-stage labels (where available).

---

## Preprocessing (implemented in scripts)

1. Load and concatenate per-subject CSVs (timestamp aligned).
2. Interpolate missing sensor values (bidirectional) and drop rows missing labels.
3. Convert raw accelerometer streams into activity counts and resample/aggregate as required.
4. Smooth/interpolate heart rate and compute windowed features (e.g., 40s std, HRV proxies).
5. Build a clock proxy from step counts / time-of-day heuristics to capture circadian signals.
6. Create sliding windows (configurable window size) and label each window (middle or last timestep depending on the experiment).
7. Standardize features and handle class imbalance with oversampling and/or class weights.
8. Save processed files to `data/processed/` for fast re-use.

Configuration parameters (window size, feature list, resampling rate, oversampling vs. class-weighting) are exposed in the training scripts.

---

## Implemented architectures

### 1) CNN–LSTM (baseline)

Conv1D backbone for local temporal feature extraction → LSTM for sequence modeling → Dense → Softmax. Used as the main baseline and for ablation comparisons.

### 2) CNN–BiLSTM–Self-Attention–PseudoQNN (advanced hybrid)

CNN backbone → BiLSTM (return sequences) → self-attention over timesteps → PseudoQNN mapping (sinusoidal/quantum-inspired nonlinearity) → classifier. The PseudoQNN is a classical, quantum-inspired mapping designed to emulate aspects of quantum feature maps in a computationally tractable form.

### 3) Quantum Neural Network (QNN) variants

Classical encoder → parameterized quantum circuit (PennyLane) as a learnable layer → classical decoder. QNN experiments are implemented as research baselines on classical simulators; several ansatz depths and encodings are provided for experimentation.

**Novelty claim:** To our knowledge, applying QNNs or PseudoQNN hybrids to wearable-derived sleep-stage classification is novel or among the earliest such investigations in this domain. This statement is conservative — please treat it as a research claim that should be verified when publishing or citing.

---

## Training & hyperparameters (typical)

* **Baseline CNN–LSTM:** Adam optimizer, categorical crossentropy, up to 100 epochs, batch size 32, early stopping (patience configurable), validation split 0.2.
* **Advanced hybrid:** Adam or AdamW (lr ≈ 1e-3), weight decay, dropout, optional Gaussian noise augmentation; typically converges faster (~50 epochs).
* **QNN experiments:** PennyLane circuits with shallow ansatz (2–4 layers), tested on classical simulators; best quantum-hybrid configurations often use a shallow ansatz because of simulator and optimization constraints.

Exact hyperparameters and training configurations are included in the respective training scripts (`cnn_lstm_attention.py`, `cnn_bilstm_pseudoqnn.py`, `train_qnn.py`).

---

## Evaluation & explainability

* Implemented metrics: per-class precision/recall/F1, accuracy, macro F1, Cohen’s Kappa, confusion matrices (raw and normalized), and macro AUC where applicable.
* Explainability: temporal attribution techniques (SHAP, timestep attribution, Grad-CAM-style visualizations) are used to inspect which timesteps and features drive predictions. Visual outputs are saved under `results/`.

---

## Reproduction instructions

1. Create and activate environment:

```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Prepare data:

* Download or clone the raw files from the source repo ([https://github.com/ojwalch/sleep_accel](https://github.com/ojwalch/sleep_accel)) and place the CSVs into `data/raw/`, or place preprocessed files into `data/processed/`.

3. Train and evaluate baseline (example):

```bash
python cnn_lstm_attention.py
```

4. Train and evaluate advanced hybrid:

```bash
python cnn_bilstm_pseudoqnn.py
```

5. Run QNN experiments (simulators):

```bash
python train_qnn.py
# or python 6_Qubits_4_layers.py depending on the script naming in this repo
```

6. Generate evaluation plots:

```bash
python evaluation.py
# figures will be saved to results/
```

> Training may take several hours on CPU. GPU (CUDA) is strongly recommended for deep models. QNN simulations can be slow for larger qubit counts; to run quick tests, use fewer qubits or a smaller dataset sample.

---

## Models / files policy

* Trained model weights are not included in this repository to keep it lightweight and reproducible.
* After running the training scripts, model weights will be saved to `models/` (for example: `models/cnn_lstm_best.h5`).
* If a prebuilt model release is provided later, it will be documented in `models/README.md` with download instructions.

---

## Results (high-level)

* Baseline and advanced hybrid models achieve strong performance on wearable-derived signals; per-paper numeric tables, ablation studies, and class-wise metrics are documented in the attached papers and the `results/` folder.
* QNN experiments provide exploratory baselines on classical simulators; some shallow quantum-hybrid configurations produced competitive results in simulator settings. See `results/` for plots and the QNN paper for detailed numbers and discussion.

---

## Resume / LinkedIn bullets (examples)

* Developed a CNN–LSTM pipeline for 5-class sleep-stage classification using wearable signals; implemented preprocessing, sliding-window feature engineering, and end-to-end training/evaluation.
* Designed and evaluated a CNN–BiLSTM–Self-Attention–PseudoQNN hybrid that improved per-class recall and overall classification metrics over the baseline.
* Explored quantum-hybrid QNN baselines (PennyLane-backed simulators) for sleep-stage classification; to our knowledge, these QNN experiments are among the earliest applications of QNNs to wearable sleep-stage classification.

---

## Limitations & ethics

* Wearable-derived features are surrogate signals and are not a replacement for clinical PSG diagnostics. Generalization to clinical populations requires additional validation.
* QNN experiments were executed on classical simulators; a practical quantum advantage has not been demonstrated and the quantum models are exploratory.
* Any clinical or health-related application requires ethical review, domain expert validation, and regulatory considerations before deployment.

---

## Citation

If you use this repository, the attached methodologies, or the underlying data, please cite the accompanying papers and acknowledge the data source repository ([https://github.com/ojwalch/sleep_accel](https://github.com/ojwalch/sleep_accel)) where raw files were obtained.