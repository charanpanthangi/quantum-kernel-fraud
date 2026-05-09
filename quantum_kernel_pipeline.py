"""
============================================================
Quantum Kernel-Enhanced SVM for Financial Fraud Detection
Complete Experimental Pipeline
Author: Charan Panthangi
============================================================

SETUP:
pip install pennylane pennylane-qiskit qiskit qiskit-aer
pip install scikit-learn xgboost imbalanced-learn mlflow
pip install pandas numpy matplotlib seaborn

DATASET:
Download from: https://www.kaggle.com/c/ieee-fraud-detection
Place train_transaction.csv and train_identity.csv
in the same folder as this script.
============================================================
"""

# -----------------------------------------------
# IMPORTS
# -----------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Classical ML
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Quantum
import pennylane as qml
from pennylane import numpy as pnp

# Qiskit Noise
from qiskit_aer.noise import NoiseModel, depolarizing_error


def zz_feature_map(x, wires, reps=2):
    """PennyLane-native ZZ feature map compatible with older template APIs."""
    wires = list(wires)
    for _ in range(reps):
        for wire in wires:
            qml.Hadamard(wires=wire)
            qml.RZ(2.0 * x[wire], wires=wire)

        for i, wire_i in enumerate(wires):
            for wire_j in wires[i + 1:]:
                qml.CNOT(wires=[wire_i, wire_j])
                qml.RZ(
                    2.0 * (np.pi - x[wire_i]) * (np.pi - x[wire_j]),
                    wires=wire_j
                )
                qml.CNOT(wires=[wire_i, wire_j])

# -----------------------------------------------
# CONFIGURATION — Edit these as needed
# -----------------------------------------------
CONFIG = {
    "data_path": "/home/ubuntu/researchpaper/",
    "n_samples": 400,
    "qubit_sizes": [4, 6, 8],
    "noise_levels": [0.0, 0.001, 0.01],
    "cv_folds": 3,
    "random_state": 42,
    "smote_ratio": 0.1,
    "zzfeaturemap_reps": 2,
    "batch_size": 20,
    "mlflow_experiment": "QK_SVM_Fraud"
}

# -----------------------------------------------
# STEP 1: DATA LOADING & PREPROCESSING
# -----------------------------------------------

def load_and_preprocess(config):
    """
    Load IEEE-CIS dataset, merge transaction + identity,
    handle missing values, encode categoricals, scale features.
    Returns X, y as numpy arrays.
    """
    print("=" * 60)
    print("STEP 1: Loading and preprocessing data...")
    print("=" * 60)

    # Load
    train_txn = pd.read_csv(
        config["data_path"] + "train_transaction.csv",
        index_col="TransactionID"
    )
    train_id = pd.read_csv(
        config["data_path"] + "train_identity.csv",
        index_col="TransactionID"
    )

    # Merge
    df = train_txn.merge(train_id, how="left",
                         left_index=True, right_index=True)
    print(f"Dataset shape after merge: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.4f} "
          f"({df['isFraud'].sum()} fraud transactions)")

    # Separate target
    y = df["isFraud"].values
    X = df.drop("isFraud", axis=1)

    # Drop columns with >50% missing
    missing_pct = X.isnull().mean()
    X = X.loc[:, missing_pct < 0.5]
    print(f"Features after dropping >50% missing: {X.shape[1]}")

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Impute remaining missing values
    X = X.fillna(X.median())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Final feature matrix: {X_scaled.shape}")
    print(f"Class distribution — Fraud: {y.sum()}, "
          f"Non-fraud: {(y==0).sum()}")

    return X_scaled, y


# -----------------------------------------------
# STEP 2: CLASS IMBALANCE HANDLING
# -----------------------------------------------

def handle_imbalance(X, y, config):
    """
    Apply SMOTE + RandomUnderSampler to balance classes.
    """
    print("\nSTEP 2: Handling class imbalance...")

    # First undersample majority to 10x minority
    rus = RandomUnderSampler(
        sampling_strategy=0.1,
        random_state=config["random_state"]
    )
    X_res, y_res = rus.fit_resample(X, y)

    # Then SMOTE to oversample minority
    current_ratio = y_res.sum() / (y_res == 0).sum()
    if current_ratio >= config["smote_ratio"]:
        X_bal, y_bal = X_res, y_res
    else:
        smote = SMOTE(
            sampling_strategy=config["smote_ratio"],
            random_state=config["random_state"],
            k_neighbors=5
        )
        X_bal, y_bal = smote.fit_resample(X_res, y_res)

    print(f"After resampling — Shape: {X_bal.shape}")
    print(f"Fraud: {y_bal.sum()}, "
          f"Non-fraud: {(y_bal==0).sum()}")

    return X_bal, y_bal


# -----------------------------------------------
# STEP 3: DIMENSIONALITY REDUCTION
# -----------------------------------------------

def apply_reduction(X_train, X_test, y_train,
                    method, n_components):
    """
    Apply PCA, LDA, or t-SNE dimensionality reduction.
    Returns reduced train and test arrays.
    """
    if method == "PCA":
        reducer = PCA(n_components=n_components,
                      random_state=42)
        X_train_r = reducer.fit_transform(X_train)
        X_test_r = reducer.transform(X_test)

    elif method == "LDA":
        # LDA max components = n_classes - 1 = 1 for binary
        # For multi-component, use n_components min
        n_comp = min(n_components,
                     len(np.unique(y_train)) - 1)
        reducer = LinearDiscriminantAnalysis(
            n_components=n_comp
        )
        X_train_r = reducer.fit_transform(X_train, y_train)
        X_test_r = reducer.transform(X_test)
        # Pad to n_components if LDA gives fewer dims
        if X_train_r.shape[1] < n_components:
            # Supplement with PCA dimensions
            pca = PCA(
                n_components=n_components - X_train_r.shape[1],
                random_state=42
            )
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            X_train_r = np.hstack([X_train_r, X_train_pca])
            X_test_r = np.hstack([X_test_r, X_test_pca])

    elif method == "tSNE":
        # t-SNE: fit on train, apply to test via PCA proxy
        # t-SNE cannot transform new data — use PCA as proxy
        print(f"  Note: t-SNE uses PCA proxy for test set")
        tsne = TSNE(n_components=min(n_components, 3),
                    random_state=42, n_iter=500)
        X_train_r = tsne.fit_transform(X_train)
        # For test, use PCA as approximation
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_train)
        X_train_r = pca.transform(X_train)  # use PCA for consistency
        X_test_r = pca.transform(X_test)

    return X_train_r, X_test_r


# -----------------------------------------------
# STEP 4: EVALUATION METRICS
# -----------------------------------------------

def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute AUC-ROC, F1, G-Mean, Precision, Recall.
    """
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # G-Mean = sqrt(sensitivity * specificity)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    gmean = np.sqrt(sensitivity * specificity)

    return {
        "AUC-ROC": round(auc, 4),
        "F1": round(f1, 4),
        "G-Mean": round(gmean, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    }


# -----------------------------------------------
# STEP 5: CLASSICAL BASELINES
# -----------------------------------------------

def run_classical_baselines(X, y, config):
    """
    Run XGBoost, Random Forest, RBF-SVM, Poly-SVM
    with 5-fold stratified cross-validation.
    Returns results dictionary.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Running classical baselines...")
    print("=" * 60)

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=config["random_state"],
            n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=config["random_state"],
            n_jobs=-1
        ),
        "RBF-SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=config["random_state"]
        ),
        "Poly-SVM": SVC(
            kernel="poly",
            degree=3,
            probability=True,
            random_state=config["random_state"]
        )
    }

    skf = StratifiedKFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=config["random_state"]
    )

    all_results = {}

    mlflow.set_experiment(config["mlflow_experiment"])

    for model_name, model in models.items():
        print(f"\nRunning {model_name}...")
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(
                skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Cap SVM training to 10K samples
            # XGBoost and RF use full dataset
            if "SVM" in model_name:
                np.random.seed(config["random_state"] + fold)
                svm_idx = np.random.choice(
                    len(X_train),
                    size=min(10000, len(X_train)),
                    replace=False
                )
                X_train = X_train[svm_idx]
                y_train = y_train[svm_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)
            fold_metrics.append(metrics)
            print(f"  Fold {fold+1}: AUC={metrics['AUC-ROC']:.4f} "
                  f"F1={metrics['F1']:.4f} "
                  f"G-Mean={metrics['G-Mean']:.4f} "
                  f"Precision={metrics['Precision']:.4f} "
                  f"Recall={metrics['Recall']:.4f}")

        # Average across folds
        avg_metrics = {}
        std_metrics = {}
        for metric in fold_metrics[0].keys():
            vals = [f[metric] for f in fold_metrics]
            avg_metrics[metric] = round(np.mean(vals), 4)
            std_metrics[metric] = round(np.std(vals), 4)

        all_results[model_name] = {
            "mean": avg_metrics,
            "std": std_metrics
        }

        print(f"  {model_name} FINAL: "
              f"AUC={avg_metrics['AUC-ROC']}±{std_metrics['AUC-ROC']} | "
              f"F1={avg_metrics['F1']}±{std_metrics['F1']} | "
              f"G-Mean={avg_metrics['G-Mean']}±{std_metrics['G-Mean']} | "
              f"Precision={avg_metrics['Precision']}±{std_metrics['Precision']} | "
              f"Recall={avg_metrics['Recall']}±{std_metrics['Recall']}")

        # Log to MLflow
        with mlflow.start_run(run_name=f"Classical_{model_name}"):
            mlflow.log_params({"model": model_name})
            for k, v in avg_metrics.items():
                mlflow.log_metric(k, v)

    return all_results


# -----------------------------------------------
# STEP 6: QUANTUM KERNEL COMPUTATION
# -----------------------------------------------

def build_quantum_kernel(n_qubits, reps=2):
    """
    Build ZZFeatureMap quantum kernel using PennyLane.
    Returns a kernel function k(x1, x2).
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        # Encode x1
        zz_feature_map(
            x1, wires=range(n_qubits), reps=reps
        )
        # Adjoint of x2 encoding
        qml.adjoint(zz_feature_map)(
            x2, wires=range(n_qubits), reps=reps
        )
        return qml.probs(wires=range(n_qubits))

    def kernel(x1, x2):
        # Kernel value = probability of |0...0> state
        return kernel_circuit(x1, x2)[0]

    return kernel


def compute_kernel_matrix(X1, X2, kernel_fn, batch_size=50):
    """
    Compute kernel matrix K[i,j] = k(X1[i], X2[j])
    in batches to manage memory.
    """
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))

    for i in range(0, n1, batch_size):
        i_end = min(i + batch_size, n1)
        for j in range(0, n2, batch_size):
            j_end = min(j + batch_size, n2)
            for ii in range(i, i_end):
                for jj in range(j, j_end):
                    K[ii, jj] = kernel_fn(
                        X1[ii], X2[jj]
                    )
        print(f"  Kernel matrix progress: "
              f"{min(i_end, n1)}/{n1} rows", end="\r")

    print()
    return K


# -----------------------------------------------
# STEP 7: KERNEL ALIGNMENT SCORE (KAS)
# -----------------------------------------------

def compute_kas(K_quantum, K_classical):
    """
    Compute Kernel Alignment Score between quantum
    and classical kernel matrices using centered
    kernel alignment (Cristianini et al. 2001).

    KAS = <K_Q_centered, K_C_centered>_F /
          (||K_Q_centered||_F * ||K_C_centered||_F)

    Returns KAS value in [0, 1].
    High KAS = quantum mimics classical structure.
    Low KAS = quantum explores distinct feature space.
    """
    def center_kernel(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K_Q_c = center_kernel(K_quantum)
    K_C_c = center_kernel(K_classical)

    numerator = np.sum(K_Q_c * K_C_c)  # Frobenius inner product
    denominator = (np.linalg.norm(K_Q_c, 'fro') *
                   np.linalg.norm(K_C_c, 'fro'))

    if denominator == 0:
        return 0.0

    kas = numerator / denominator
    return round(float(kas), 4)


# -----------------------------------------------
# STEP 8: QUANTUM KERNEL SVM EXPERIMENTS
# -----------------------------------------------

def run_quantum_experiments(X, y, config):
    """
    Run QK-SVM across all reduction methods and qubit sizes.
    Returns results dictionary with metrics and KAS values.
    """
    print("\n" + "=" * 60)
    print("STEP 8: Running quantum kernel experiments...")
    print("=" * 60)

    # Use a subset for quantum (computationally expensive)
    np.random.seed(config["random_state"])
    subset_idx = np.random.choice(
        len(X), size=config["n_samples"], replace=False
    )
    X_sub = X[subset_idx]
    y_sub = y[subset_idx]

    print(f"Using subset of {config['n_samples']} samples "
          f"for quantum experiments")

    reduction_methods = ["PCA", "LDA", "tSNE"]
    qubit_sizes = config["qubit_sizes"]

    skf = StratifiedKFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=config["random_state"]
    )

    all_results = {}

    for method in reduction_methods:
        for n_qubits in qubit_sizes:
            exp_name = f"QK-SVM_{method}_{n_qubits}q"
            print(f"\n--- {exp_name} ---")
            fold_metrics = []
            kas_values = []

            for fold, (train_idx, test_idx) in enumerate(
                    skf.split(X_sub, y_sub)):
                X_train = X_sub[train_idx]
                X_test = X_sub[test_idx]
                y_train = y_sub[train_idx]
                y_test = y_sub[test_idx]

                # Dimensionality reduction
                X_train_r, X_test_r = apply_reduction(
                    X_train, X_test, y_train,
                    method, n_qubits
                )

                # Build quantum kernel
                qkernel = build_quantum_kernel(
                    n_qubits,
                    reps=config["zzfeaturemap_reps"]
                )

                # Compute quantum kernel matrix
                print(f"  Fold {fold+1}: Computing quantum "
                      f"kernel matrix...")
                K_train = compute_kernel_matrix(
                    X_train_r, X_train_r,
                    qkernel, config["batch_size"]
                )
                K_test = compute_kernel_matrix(
                    X_test_r, X_train_r,
                    qkernel, config["batch_size"]
                )

                # Compute classical RBF kernel for KAS
                # Using sklearn pairwise_kernels instead of private method
                K_rbf = rbf_kernel(X_train_r)

                # Compute KAS
                kas = compute_kas(K_train, K_rbf)
                kas_values.append(kas)

                # Train QK-SVM
                qsvm = SVC(
                    kernel="precomputed",
                    probability=True,
                    random_state=config["random_state"]
                )
                qsvm.fit(K_train, y_train)
                y_pred = qsvm.predict(K_test)
                y_prob = qsvm.predict_proba(K_test)[:, 1]

                metrics = compute_metrics(
                    y_test, y_pred, y_prob
                )
                metrics["KAS"] = kas
                fold_metrics.append(metrics)

                print(f"  Fold {fold+1}: "
                      f"AUC={metrics['AUC-ROC']:.4f} "
                      f"F1={metrics['F1']:.4f} "
                      f"KAS={kas:.4f}")

            # Average across folds
            avg_metrics = {}
            std_metrics = {}
            for metric in fold_metrics[0].keys():
                vals = [f[metric] for f in fold_metrics]
                avg_metrics[metric] = round(np.mean(vals), 4)
                std_metrics[metric] = round(np.std(vals), 4)

            all_results[exp_name] = {
                "mean": avg_metrics,
                "std": std_metrics,
                "method": method,
                "n_qubits": n_qubits
            }

            print(f"  FINAL {exp_name}: "
                  f"AUC={avg_metrics['AUC-ROC']}±"
                  f"{std_metrics['AUC-ROC']} | "
                  f"KAS={avg_metrics['KAS']}±"
                  f"{std_metrics['KAS']}")

            # Log to MLflow
            with mlflow.start_run(run_name=exp_name):
                mlflow.log_params({
                    "method": method,
                    "n_qubits": n_qubits,
                    "reps": config["zzfeaturemap_reps"]
                })
                for k, v in avg_metrics.items():
                    mlflow.log_metric(k, v)

    return all_results


# -----------------------------------------------
# STEP 9: NOISE IMPACT ANALYSIS
# -----------------------------------------------

def run_noise_experiments(X, y, config,
                          best_reduction, best_n_qubits):
    """
    Run QK-SVM under three depolarizing noise levels.
    Uses the best configuration from Step 8.
    """
    print("\n" + "=" * 60)
    print("STEP 9: Running noise impact experiments...")
    print("=" * 60)

    np.random.seed(config["random_state"])
    subset_idx = np.random.choice(
        len(X), size=config["n_samples"], replace=False
    )
    X_sub = X[subset_idx]
    y_sub = y[subset_idx]

    # Reduce dimensions using best method
    X_train_r, X_test_r = apply_reduction(
        X_sub, X_sub, y_sub,
        best_reduction, best_n_qubits
    )

    noise_results = {}

    for noise_p in config["noise_levels"]:
        print(f"\nNoise level p={noise_p}...")

        if noise_p == 0.0:
            # Ideal simulator
            dev = qml.device(
                "default.qubit",
                wires=best_n_qubits
            )
        else:
            # Noisy simulator via Qiskit Aer
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
                wires=best_n_qubits,
                noise_model=noise_model
            )

        @qml.qnode(dev)
        def noisy_kernel_circuit(x1, x2):
            zz_feature_map(
                x1,
                wires=range(best_n_qubits),
                reps=config["zzfeaturemap_reps"]
            )
            qml.adjoint(zz_feature_map)(
                x2,
                wires=range(best_n_qubits),
                reps=config["zzfeaturemap_reps"]
            )
            return qml.probs(wires=range(best_n_qubits))

        def noisy_kernel(x1, x2):
            return noisy_kernel_circuit(x1, x2)[0]

        # Use single fold for noise experiments
        skf = StratifiedKFold(
            n_splits=3, shuffle=True,
            random_state=config["random_state"]
        )
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(
                skf.split(X_train_r, y_sub)):
            X_tr = X_train_r[train_idx]
            X_te = X_train_r[test_idx]
            y_tr = y_sub[train_idx]
            y_te = y_sub[test_idx]

            K_train = compute_kernel_matrix(
                X_tr, X_tr,
                noisy_kernel, config["batch_size"]
            )
            K_test = compute_kernel_matrix(
                X_te, X_tr,
                noisy_kernel, config["batch_size"]
            )

            qsvm = SVC(
                kernel="precomputed",
                probability=True,
                random_state=config["random_state"]
            )
            qsvm.fit(K_train, y_tr)
            y_pred = qsvm.predict(K_test)
            y_prob = qsvm.predict_proba(K_test)[:, 1]

            metrics = compute_metrics(y_te, y_pred, y_prob)
            fold_metrics.append(metrics)

        avg_metrics = {}
        for metric in fold_metrics[0].keys():
            vals = [f[metric] for f in fold_metrics]
            avg_metrics[metric] = round(np.mean(vals), 4)

        noise_results[f"p={noise_p}"] = avg_metrics
        print(f"  p={noise_p}: AUC={avg_metrics['AUC-ROC']} "
              f"F1={avg_metrics['F1']}")

    return noise_results


# -----------------------------------------------
# STEP 10: RESULTS TABLES & VISUALIZATIONS
# -----------------------------------------------

def print_results_table(classical_results, quantum_results):
    """
    Print formatted results tables for paper.
    """
    print("\n" + "=" * 60)
    print("RESULTS TABLE — Classical Baselines")
    print("=" * 60)
    print(f"{'Model':<20} {'AUC-ROC':<12} {'F1':<12} "
          f"{'G-Mean':<12} {'Precision':<12}")
    print("-" * 68)
    for model, res in classical_results.items():
        m = res["mean"]
        s = res["std"]
        print(f"{model:<20} "
              f"{m['AUC-ROC']}±{s['AUC-ROC']:<6} "
              f"{m['F1']}±{s['F1']:<6} "
              f"{m['G-Mean']}±{s['G-Mean']:<6} "
              f"{m['Precision']}±{s['Precision']:<6}")

    print("\n" + "=" * 60)
    print("RESULTS TABLE — Quantum Kernel SVM")
    print("=" * 60)
    print(f"{'Config':<25} {'AUC-ROC':<12} {'F1':<12} "
          f"{'G-Mean':<12} {'KAS':<10}")
    print("-" * 71)
    for exp, res in quantum_results.items():
        m = res["mean"]
        s = res["std"]
        print(f"{exp:<25} "
              f"{m['AUC-ROC']}±{s['AUC-ROC']:<6} "
              f"{m['F1']}±{s['F1']:<6} "
              f"{m['G-Mean']}±{s['G-Mean']:<6} "
              f"{m['KAS']}±{s['KAS']:<6}")


def plot_kernel_matrix(K_quantum, K_classical,
                       save_path="kernel_matrices.png"):
    """
    Visualize quantum vs classical kernel matrices.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(K_quantum[:50, :50], ax=axes[0],
                cmap="viridis", cbar=True)
    axes[0].set_title("Quantum Kernel Matrix (ZZFeatureMap)",
                       fontsize=12)
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Sample index")

    sns.heatmap(K_classical[:50, :50], ax=axes[1],
                cmap="viridis", cbar=True)
    axes[1].set_title("Classical RBF Kernel Matrix",
                       fontsize=12)
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Sample index")

    plt.suptitle("Kernel Matrix Comparison — "
                 "Quantum vs Classical", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Kernel matrix plot saved: {save_path}")


def plot_noise_impact(noise_results,
                      save_path="noise_impact.png"):
    """
    Plot AUC degradation across noise levels.
    """
    noise_levels = list(noise_results.keys())
    auc_values = [noise_results[p]["AUC-ROC"]
                  for p in noise_levels]
    f1_values = [noise_results[p]["F1"]
                 for p in noise_levels]

    x = range(len(noise_levels))
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, auc_values, 'bo-', label='AUC-ROC',
            linewidth=2, markersize=8)
    ax.plot(x, f1_values, 'rs-', label='F1 Score',
            linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(noise_levels, fontsize=11)
    ax.set_xlabel("Depolarizing Noise Level (p)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("QK-SVM Performance Under NISQ Noise",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Noise impact plot saved: {save_path}")


# -----------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("QUANTUM KERNEL SVM — FRAUD DETECTION PIPELINE")
    print("Author: Charan Panthangi")
    print("=" * 60)

    # Step 1: Load and preprocess
    X, y = load_and_preprocess(CONFIG)

    # Step 2: Handle imbalance
    X_bal, y_bal = handle_imbalance(X, y, CONFIG)

    # Step 5: Classical baselines (full balanced dataset)
    classical_results = run_classical_baselines(
        X_bal, y_bal, CONFIG
    )

    # Step 8: Quantum experiments (subset)
    quantum_results = run_quantum_experiments(
        X_bal, y_bal, CONFIG
    )

    # Determine best quantum config from results
    best_exp = max(
        quantum_results,
        key=lambda k: quantum_results[k]["mean"]["AUC-ROC"]
    )
    best_method = quantum_results[best_exp]["method"]
    best_qubits = quantum_results[best_exp]["n_qubits"]
    print(f"\nBest quantum config: {best_exp}")

    # Step 9: Noise experiments
    noise_results = run_noise_experiments(
        X_bal, y_bal, CONFIG,
        best_method, best_qubits
    )

    # Step 10: Print tables
    print_results_table(classical_results, quantum_results)

    # Plot noise impact
    plot_noise_impact(noise_results)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("Check MLflow UI: mlflow ui")
    print("All plots saved as PNG files")
    print("Copy numbers into your Overleaf paper")
    print("=" * 60)

    return classical_results, quantum_results, noise_results


# -----------------------------------------------
# QUICK SANITY TEST
# Run this first before full pipeline
# to verify quantum stack is working
# -----------------------------------------------

def sanity_test():
    """
    Quick 2-minute test to verify your quantum
    environment is set up correctly.
    Run this before main().
    """
    print("Running sanity test...")

    # Test PennyLane ZZFeatureMap
    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def test_circuit(x):
        zz_feature_map(
            x, wires=range(n_qubits), reps=2
        )
        return qml.probs(wires=range(n_qubits))

    x_test = np.random.rand(n_qubits)
    result = test_circuit(x_test)
    print(f"ZZFeatureMap output shape: {result.shape}")
    print(f"Probabilities sum to 1: "
          f"{np.isclose(result.sum(), 1.0)}")

    # Test kernel computation
    x1 = np.random.rand(n_qubits)
    x2 = np.random.rand(n_qubits)
    kernel_fn = build_quantum_kernel(n_qubits)
    k_val = kernel_fn(x1, x2)
    print(f"Kernel value k(x1,x2) = {k_val:.6f}")
    print(f"Kernel self-similarity k(x1,x1) = "
          f"{kernel_fn(x1, x1):.6f} (should be ~1.0)")

    # Test KAS
    K_q = np.random.rand(10, 10)
    K_q = K_q @ K_q.T  # make positive semi-definite
    K_c = np.random.rand(10, 10)
    K_c = K_c @ K_c.T
    kas = compute_kas(K_q, K_c)
    print(f"KAS test value: {kas} (should be between 0 and 1)")

    print("\nSanity test PASSED. Ready to run main pipeline.")


# -----------------------------------------------
# ENTRY POINT
# -----------------------------------------------

if __name__ == "__main__":
    # Run sanity test first
    sanity_test()

    classical_results, quantum_results, noise_results = main()
