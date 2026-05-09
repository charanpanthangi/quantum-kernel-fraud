"""
============================================================
NOISE-ONLY EXPERIMENTS — All Configurations
Loads exact same subset from Step 8 via subset_idx.npy
Author: Charan Panthangi
============================================================
Runs Step 9 noise experiments for:
- PCA: 4q, 6q, 8q
- LDA: 4q, 6q, 8q
- tSNE: 4q, 6q, 8q
At noise levels: p=0.0, p=0.001, p=0.01
Uses SAME 400 samples as Step 8 via subset_idx.npy
============================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pennylane as qml

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
DATA_PATH = "/home/ubuntu/researchpaper/"
SUBSET_IDX_PATH = "/home/ubuntu/researchpaper/subset_idx.npy"
RANDOM_STATE = 42
CV_FOLDS = 3
REPS = 2
NOISE_LEVELS = [0.0, 0.001, 0.01]
CONFIGS = [
    ("PCA", 4), ("PCA", 6), ("PCA", 8),
    ("LDA", 4), ("LDA", 6), ("LDA", 8),
    ("tSNE", 4), ("tSNE", 6), ("tSNE", 8),
]

# -----------------------------------------------
# ZZ FEATURE MAP
# -----------------------------------------------
def zz_feature_map(x, wires, reps=2):
    wires = list(wires)
    for _ in range(reps):
        for wire in wires:
            qml.Hadamard(wires=wire)
            qml.RZ(2.0 * x[wire], wires=wire)
        for i, wire_i in enumerate(wires):
            for wire_j in wires[i + 1:]:
                qml.CNOT(wires=[wire_i, wire_j])
                qml.RZ(
                    2.0 * (np.pi - x[wire_i]) *
                    (np.pi - x[wire_j]),
                    wires=wire_j
                )
                qml.CNOT(wires=[wire_i, wire_j])

# -----------------------------------------------
# STEP 1: LOAD DATA
# -----------------------------------------------
print("=" * 60)
print("Loading and preprocessing data...")
print("=" * 60)

train_txn = pd.read_csv(
    DATA_PATH + "train_transaction.csv",
    index_col="TransactionID"
)
train_id = pd.read_csv(
    DATA_PATH + "train_identity.csv",
    index_col="TransactionID"
)
df = train_txn.merge(
    train_id, how="left",
    left_index=True, right_index=True
)

y = df["isFraud"].values
X = df.drop("isFraud", axis=1)
X = X.loc[:, X.isnull().mean() < 0.5]

for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X = X.fillna(X.median())
X = StandardScaler().fit_transform(X)
print(f"Data loaded: {X.shape}")

# -----------------------------------------------
# STEP 2: HANDLE IMBALANCE
# -----------------------------------------------
print("Handling imbalance...")
rus = RandomUnderSampler(
    sampling_strategy=0.1,
    random_state=RANDOM_STATE
)
X_res, y_res = rus.fit_resample(X, y)

current_ratio = y_res.sum() / (y_res == 0).sum()
if current_ratio >= 0.1:
    X_bal, y_bal = X_res, y_res
else:
    smote = SMOTE(
        sampling_strategy=0.1,
        random_state=RANDOM_STATE,
        k_neighbors=5
    )
    X_bal, y_bal = smote.fit_resample(X_res, y_res)

print(f"After resampling: {X_bal.shape}")

# -----------------------------------------------
# LOAD EXACT SAME SUBSET FROM STEP 8
# -----------------------------------------------
print(f"\nLoading subset indices from {SUBSET_IDX_PATH}...")
subset_idx = np.load(SUBSET_IDX_PATH)
X_sub = X_bal[subset_idx]
y_sub = y_bal[subset_idx]
print(f"Loaded {len(subset_idx)} samples — "
      f"same as Step 8 ✅")
print(f"Fraud in subset: {y_sub.sum()}, "
      f"Non-fraud: {(y_sub==0).sum()}")

# -----------------------------------------------
# DIMENSIONALITY REDUCTION
# -----------------------------------------------
def apply_reduction(X_train, X_test, y_train,
                    method, n_components):
    if method == "PCA":
        r = PCA(n_components=n_components, random_state=42)
        return r.fit_transform(X_train), r.transform(X_test)

    elif method == "LDA":
        n_comp = min(n_components,
                     len(np.unique(y_train)) - 1)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        X_lda_tr = lda.fit_transform(X_train, y_train)
        X_lda_te = lda.transform(X_test)
        if X_lda_tr.shape[1] < n_components:
            pca = PCA(
                n_components=n_components - X_lda_tr.shape[1],
                random_state=42
            )
            X_pca_tr = pca.fit_transform(X_train)
            X_pca_te = pca.transform(X_test)
            return (np.hstack([X_lda_tr, X_pca_tr]),
                    np.hstack([X_lda_te, X_pca_te]))
        return X_lda_tr, X_lda_te

    elif method == "tSNE":
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(X_train), pca.transform(X_test)

# -----------------------------------------------
# SEQUENTIAL KERNEL COMPUTATION
# Safe — no thread conflicts
# -----------------------------------------------
def build_kernel(n_qubits, noise_p=0.0):
    if noise_p == 0.0:
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        from qiskit_aer.noise import (
            NoiseModel, depolarizing_error
        )
        noise_model = NoiseModel()
        error_1q = depolarizing_error(noise_p, 1)
        error_2q = depolarizing_error(noise_p, 2)
        noise_model.add_all_qubit_quantum_error(
            error_1q, ['u1', 'u2', 'u3']
        )
        noise_model.add_all_qubit_quantum_error(
            error_2q, ['cx']
        )
        dev = qml.device(
            "qiskit.aer",
            wires=n_qubits,
            noise_model=noise_model
        )

    @qml.qnode(dev)
    def circuit(x1, x2):
        zz_feature_map(x1, wires=range(n_qubits), reps=REPS)
        qml.adjoint(zz_feature_map)(
            x2, wires=range(n_qubits), reps=REPS
        )
        return qml.probs(wires=range(n_qubits))

    def kernel(x1, x2):
        return circuit(x1, x2)[0]

    return kernel


def kernel_matrix(X1, X2, kfn, batch_size=20):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in range(0, n1, batch_size):
        i_end = min(i + batch_size, n1)
        for ii in range(i, i_end):
            for jj in range(n2):
                K[ii, jj] = kfn(X1[ii], X2[jj])
        print(f"  Progress: {i_end}/{n1} rows", end="\r")
    print()
    return K


# -----------------------------------------------
# METRICS
# -----------------------------------------------
def get_metrics(y_true, y_pred, y_prob):
    return (
        round(roc_auc_score(y_true, y_prob), 4),
        round(f1_score(y_true, y_pred, zero_division=0), 4)
    )


# -----------------------------------------------
# MAIN: NOISE EXPERIMENTS
# -----------------------------------------------
print("\n" + "=" * 60)
print("STEP 9: NOISE EXPERIMENTS — ALL CONFIGS")
print("Same 400 samples as Step 8")
print("=" * 60)

all_results = {}
skf = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

for method, n_qubits in CONFIGS:
    config_name = f"{method}_{n_qubits}q"
    print(f"\n{'=' * 50}")
    print(f"Config: {config_name}")
    print(f"{'=' * 50}")

    noise_results = {}

    for noise_p in NOISE_LEVELS:
        print(f"\nNoise level p={noise_p}...")
        fold_aucs = []
        fold_f1s = []

        kfn = build_kernel(n_qubits, noise_p)

        for fold_idx, (train_idx, test_idx) in enumerate(
                skf.split(X_sub, y_sub)):
            X_tr = X_sub[train_idx]
            X_te = X_sub[test_idx]
            y_tr = y_sub[train_idx]
            y_te = y_sub[test_idx]

            X_tr_r, X_te_r = apply_reduction(
                X_tr, X_te, y_tr, method, n_qubits
            )

            print(f"  Fold {fold_idx+1} train kernel...")
            K_train = kernel_matrix(X_tr_r, X_tr_r, kfn)
            print(f"  Fold {fold_idx+1} test kernel...")
            K_test = kernel_matrix(X_te_r, X_tr_r, kfn)

            svm = SVC(
                kernel="precomputed",
                probability=True,
                random_state=RANDOM_STATE
            )
            svm.fit(K_train, y_tr)
            y_pred = svm.predict(K_test)
            y_prob = svm.predict_proba(K_test)[:, 1]

            auc, f1 = get_metrics(y_te, y_pred, y_prob)
            fold_aucs.append(auc)
            fold_f1s.append(f1)
            print(f"  Fold {fold_idx+1}: AUC={auc} F1={f1}")

        avg_auc = round(np.mean(fold_aucs), 4)
        std_auc = round(np.std(fold_aucs), 4)
        avg_f1 = round(np.mean(fold_f1s), 4)

        noise_results[f"p={noise_p}"] = {
            "AUC": avg_auc,
            "std": std_auc,
            "F1": avg_f1
        }

        print(f"  p={noise_p} FINAL: "
              f"AUC={avg_auc}±{std_auc} | F1={avg_f1}")

    all_results[config_name] = noise_results

# -----------------------------------------------
# FINAL RESULTS TABLE
# -----------------------------------------------
print("\n" + "=" * 60)
print("FINAL NOISE RESULTS TABLE — ALL CONFIGS")
print("=" * 60)
print(f"{'Config':<12} {'p=0.0':<20} "
      f"{'p=0.001':<20} {'p=0.01':<20}")
print("-" * 72)

for config_name, noise_res in all_results.items():
    r0 = noise_res.get('p=0.0', {})
    r1 = noise_res.get('p=0.001', {})
    r2 = noise_res.get('p=0.01', {})
    print(f"{config_name:<12} "
          f"AUC={r0.get('AUC','N/A')}±"
          f"{r0.get('std','N/A'):<8} "
          f"AUC={r1.get('AUC','N/A')}±"
          f"{r1.get('std','N/A'):<8} "
          f"AUC={r2.get('AUC','N/A')}±"
          f"{r2.get('std','N/A'):<8}")

print("\nNOISE EXPERIMENTS COMPLETE ✅")
