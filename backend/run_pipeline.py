import argparse
import json
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")

NUMERIC_BASE_COLS = [
    "Heart Rate",
    "Body Temperature",
    "Oxygen Saturation",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Age",
    "Weight (kg)",
    "Height (m)",
]

FEATURE_COLS = [
    "Heart Rate",
    "Hour",
    "Minute",
    "Second",
    "Body Temperature",
    "Oxygen Saturation",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Age",
    "Gender_encoded",
    "Weight (kg)",
    "Height (m)",
    "Derived_HRV",
    "Derived_Pulse_Pressure",
    "Derived_BMI",
    "Derived_MAP",
]

REQUIRED_COLS = set(NUMERIC_BASE_COLS + ["Timestamp", "Gender", "Risk Category"])
MAX_ROWS_FOR_TRAINING = 40000
ARTIFACT_DIR = "artifacts"
DATA_DIR = "data"
OUTPUT_DIR = "output"
RF_MODEL_FILE = "rf_model.joblib"
SCALER_FILE = "feature_scaler.joblib"
METADATA_FILE = "pipeline_metadata.json"


def remove_outliers_iqr(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    data_clean = data.copy()
    for col in cols:
        q1 = data_clean[col].quantile(0.25)
        q3 = data_clean[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        data_clean = data_clean[(data_clean[col] >= low) & (data_clean[col] <= high)]
    return data_clean


def create_sequences(X_arr: np.ndarray, y_arr: np.ndarray, timesteps: int = 10) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X_arr) - timesteps + 1):
        Xs.append(X_arr[i : i + timesteps])
        ys.append(y_arr[i + timesteps - 1])
    return np.array(Xs), np.array(ys)


def calculate_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")


def encode_gender(series: pd.Series, gender_classes: list[str] | None = None) -> tuple[pd.Series, list[str]]:
    if gender_classes is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(series.astype(str))
        return pd.Series(encoded, index=series.index), list(encoder.classes_)

    mapping = {name: idx for idx, name in enumerate(gender_classes)}
    encoded = series.astype(str).map(mapping).fillna(-1).astype(int)
    return encoded, gender_classes


def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")

    proc = df.copy()
    for col in proc.columns:
        if proc[col].dtype == object:
            modes = proc[col].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "Unknown"
            proc[col] = proc[col].fillna(fill_value)
        else:
            proc[col] = proc[col].fillna(proc[col].mean())

    proc = remove_outliers_iqr(proc, NUMERIC_BASE_COLS)
    proc["Derived_Pulse_Pressure"] = proc["Systolic Blood Pressure"] - proc["Diastolic Blood Pressure"]
    proc["Derived_BMI"] = proc["Weight (kg)"] / (proc["Height (m)"] ** 2)
    proc["Derived_MAP"] = (proc["Systolic Blood Pressure"] + 2 * proc["Diastolic Blood Pressure"]) / 3

    parsed_ts = pd.to_datetime(proc["Timestamp"], format="%H:%M:%S.%f", errors="coerce")
    if parsed_ts.isna().all():
        parsed_ts = pd.to_datetime(proc["Timestamp"], errors="coerce")
    proc["Timestamp"] = parsed_ts

    proc = proc.sort_values("Timestamp").reset_index(drop=True)
    hr_std = proc["Heart Rate"].std()
    if pd.isna(hr_std):
        hr_std = 0.0
    proc["Derived_HRV"] = proc["Heart Rate"].rolling(window=10, min_periods=1).std().fillna(hr_std)

    proc["Hour"] = proc["Timestamp"].dt.hour.fillna(0).astype(int)
    proc["Minute"] = proc["Timestamp"].dt.minute.fillna(0).astype(int)
    proc["Second"] = proc["Timestamp"].dt.second.fillna(0).astype(int)
    proc = proc.replace([np.inf, -np.inf], np.nan)
    return proc


def save_artifacts(
    project_dir: Path,
    model: RandomForestClassifier,
    scaler: StandardScaler,
    gender_classes: list[str],
    risk_classes: list[str],
    scale_cols: list[str],
) -> None:
    artifact_path = project_dir / ARTIFACT_DIR
    artifact_path.mkdir(exist_ok=True)
    dump(model, artifact_path / RF_MODEL_FILE)
    dump(scaler, artifact_path / SCALER_FILE)
    metadata = {
        "feature_cols": FEATURE_COLS,
        "scale_cols": scale_cols,
        "gender_classes": gender_classes,
        "risk_classes": risk_classes,
    }
    (artifact_path / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_artifacts(project_dir: Path) -> tuple[RandomForestClassifier, StandardScaler, dict]:
    artifact_path = project_dir / ARTIFACT_DIR
    model = load(artifact_path / RF_MODEL_FILE)
    scaler = load(artifact_path / SCALER_FILE)
    metadata = json.loads((artifact_path / METADATA_FILE).read_text(encoding="utf-8"))
    return model, scaler, metadata


def run_test_data_process(
    project_dir: Path,
    test_data_path: Path,
    mode: str,
    model: RandomForestClassifier,
    scaler: StandardScaler,
    metadata: dict,
) -> None:
    print(f"Loading test data: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    if "Risk Category" not in test_df.columns:
        test_df["Risk Category"] = "Unknown"

    test_proc = preprocess_base(test_df)
    test_proc["Gender_encoded"], _ = encode_gender(
        test_proc["Gender"], metadata["gender_classes"]
    )
    test_proc = test_proc.dropna(subset=metadata["feature_cols"]).reset_index(drop=True)
    test_proc[metadata["scale_cols"]] = scaler.transform(test_proc[metadata["scale_cols"]])

    test_features = test_proc[metadata["feature_cols"]]
    proba = model.predict_proba(test_features)
    pred_idx = np.argmax(proba, axis=1)
    risk_classes = metadata["risk_classes"]
    pred_labels = [risk_classes[idx] for idx in pred_idx]
    test_proc["Predicted_Risk_Category"] = pred_labels
    test_proc["Predicted_Risk_Confidence"] = np.max(proba, axis=1)

    output_path = project_dir / OUTPUT_DIR
    output_path.mkdir(exist_ok=True)
    out_path = output_path / f"predictions_{test_data_path.stem}.csv"
    test_proc.to_csv(out_path, index=False)
    print(f"Prediction file generated: {out_path}")

    if mode == "evaluate" and "Risk Category" in test_df.columns:
        known_mask = test_proc["Risk Category"].isin(risk_classes)
        if known_mask.any():
            y_true = test_proc.loc[known_mask, "Risk Category"].map(
                {name: idx for idx, name in enumerate(risk_classes)}
            )
            y_pred = np.array(pred_idx)[known_mask.to_numpy()]
            print(f"Test accuracy (RF): {accuracy_score(y_true, y_pred):.4f}")
            if len(risk_classes) == 2:
                print(f"Test ROC AUC (RF): {roc_auc_score(y_true, proba[known_mask.to_numpy(), 1]):.4f}")
            else:
                print(
                    f"Test ROC AUC (RF): "
                    f"{roc_auc_score(y_true, proba[known_mask.to_numpy()], multi_class='ovr', average='weighted'):.4f}"
                )
        else:
            print("Evaluate mode skipped: no valid Risk Category labels found in test data.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pipeline and optionally process test data.")
    parser.add_argument(
        "--train-data",
        default=f"{DATA_DIR}/human_vital_signs_dataset_2024.csv",
        help="Path to training CSV relative to this script directory.",
    )
    parser.add_argument(
        "--test-data",
        default=None,
        help="Optional path to test CSV relative to this script directory.",
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "evaluate"],
        default="predict",
        help="Use evaluate only when test CSV has Risk Category labels.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run test processing using saved artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / args.train_data
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if not args.skip_train:
        print("Loading data...")
        df = pd.read_csv(data_path)
        print("Initial shape:", df.shape)
        print("Shape before outlier removal:", df.shape)
        df = preprocess_base(df)
        print("Shape after outlier removal:", df.shape)

        df["Gender_encoded"], gender_classes = encode_gender(df["Gender"])
        df = df.dropna(subset=FEATURE_COLS + ["Risk Category"]).reset_index(drop=True)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["Risk Category"])
        risk_classes = list(label_encoder.classes_)
        X = df[FEATURE_COLS].copy()
        n_classes = len(risk_classes)
        if n_classes < 2:
            raise ValueError("Risk Category needs at least two classes for model training.")

        if len(X) > MAX_ROWS_FOR_TRAINING:
            sampled = (
                X.assign(_target=y)
                .groupby("_target", group_keys=False)
                .apply(
                    lambda g: g.sample(
                        n=max(1, int(MAX_ROWS_FOR_TRAINING * (len(g) / len(X)))),
                        random_state=42,
                    )
                )
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )
            y = sampled["_target"].to_numpy(dtype=int)
            X = sampled.drop(columns="_target")
            print(f"Downsampled to {len(X)} rows for faster model training.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scale_cols = [c for c in FEATURE_COLS if c != "Gender_encoded"]
        scaler = StandardScaler()
        X_train.loc[:, scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test.loc[:, scale_cols] = scaler.transform(X_test[scale_cols])

        svm_model = SVC(probability=True, random_state=42).fit(X_train, y_train)
        rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

        timesteps = 10
        X_train_seq, y_train_seq = create_sequences(X_train.values, y_train, timesteps=timesteps)
        X_test_seq, y_test_seq = create_sequences(X_test.values, y_test, timesteps=timesteps)
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            raise ValueError(
                f"Not enough rows to create {timesteps}-step sequences after preprocessing."
            )

        lstm_model = Sequential(
            [
                LSTM(64, input_shape=(timesteps, X_train.shape[1]), return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(n_classes, activation="softmax"),
            ]
        )
        lstm_model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        lstm_model.fit(X_train_seq, y_train_seq, epochs=2, batch_size=64, verbose=1)

        svm_prob = svm_model.predict_proba(X_test)
        svm_pred = svm_model.predict(X_test)
        print(
            f"SVM acc: {accuracy_score(y_test, svm_pred):.4f}, "
            f"roc: {calculate_roc_auc(y_test, svm_prob, n_classes):.4f}"
        )

        rf_prob = rf_model.predict_proba(X_test)
        rf_pred = rf_model.predict(X_test)
        print(
            f"RandomForest acc: {accuracy_score(y_test, rf_pred):.4f}, "
            f"roc: {calculate_roc_auc(y_test, rf_prob, n_classes):.4f}"
        )

        lstm_prob = lstm_model.predict(X_test_seq)
        lstm_pred = np.argmax(lstm_prob, axis=1)
        print(
            f"LSTM acc: {accuracy_score(y_test_seq, lstm_pred):.4f}, "
            f"roc: {calculate_roc_auc(y_test_seq, lstm_prob, n_classes):.4f}"
        )

        save_artifacts(project_dir, rf_model, scaler, gender_classes, risk_classes, scale_cols)
        print(f"Artifacts saved in: {project_dir / ARTIFACT_DIR}")
    else:
        print("Skipping training and loading existing artifacts...")
        rf_model, scaler, metadata = load_artifacts(project_dir)
        if args.test_data:
            test_data_path = project_dir / args.test_data
            if not test_data_path.exists():
                raise FileNotFoundError(f"Test dataset not found: {test_data_path}")
            run_test_data_process(project_dir, test_data_path, args.mode, rf_model, scaler, metadata)
        print("Script finished successfully")
        return

    if args.test_data:
        test_data_path = project_dir / args.test_data
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {test_data_path}")
        rf_model, scaler, metadata = load_artifacts(project_dir)
        run_test_data_process(project_dir, test_data_path, args.mode, rf_model, scaler, metadata)

    print("Script finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Exception occurred:", exc)
        traceback.print_exc()
