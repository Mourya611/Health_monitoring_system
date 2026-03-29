import json
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "rf_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.joblib"
METADATA_PATH = ARTIFACTS_DIR / "pipeline_metadata.json"
PATIENTS_CSV_PATH = BASE_DIR / "patients.csv"

WINDOW_SIZE = 10
hr_window_map = {}


def load_inference_assets():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Artifacts not found. Run `python run_pipeline.py` first to create the model artifacts."
        )
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, scaler, metadata


RF_MODEL, SCALER, METADATA = load_inference_assets()


def fahrenheit_to_celsius(temp_f: float) -> float:
    return (temp_f - 32.0) * 5.0 / 9.0


def generate_blood_pressure(heart_rate: float) -> tuple[float, float]:
    rng = np.random.default_rng()
    systolic = 102 + 0.45 * (heart_rate - 60) + rng.normal(0, 4)
    diastolic = 66 + 0.28 * (heart_rate - 60) + rng.normal(0, 3)
    systolic = float(np.clip(systolic, 90, 185))
    diastolic = float(np.clip(diastolic, 55, 120))
    if diastolic >= systolic:
        diastolic = max(55.0, systolic - 12.0)
    return round(systolic, 2), round(diastolic, 2)


def encode_gender(gender: str, known_classes: list[str]) -> int:
    gender_map = {name: idx for idx, name in enumerate(known_classes)}
    return int(gender_map.get(gender, -1))


def append_patients_csv(row: dict) -> None:
    row_df = pd.DataFrame([row])
    if PATIENTS_CSV_PATH.exists():
        row_df.to_csv(PATIENTS_CSV_PATH, mode="a", index=False, header=False)
    else:
        row_df.to_csv(PATIENTS_CSV_PATH, mode="w", index=False, header=True)


def build_feature_row(payload: dict) -> tuple[dict, dict]:
    patient_id = str(payload.get("patientId", "patient1"))
    hr = float(payload["heartRate_BPM"])
    spo2_raw = float(payload["spo2_percent"])
    temperature_f = float(payload["temperature_F"])
    age = float(payload["age"])
    gender = str(payload["gender"])
    weight = float(payload["weight_kg"])
    height = float(payload["height_m"])
    status = str(payload.get("status", "UNKNOWN"))

    now = datetime.now()
    ts = now.strftime("%H:%M:%S.%f")

    window = hr_window_map.get(patient_id)
    if window is None:
        window = deque(maxlen=WINDOW_SIZE)
        hr_window_map[patient_id] = window
    window.append(hr)

    hrv = float(np.std(list(window), ddof=0)) if len(window) > 1 else 0.0
    systolic, diastolic = generate_blood_pressure(hr)
    pulse_pressure = systolic - diastolic
    bmi = weight / (height**2) if height > 0 else 0.0

    spo2_effective = min(spo2_raw, 95.0)
    map_raw = (systolic + 2 * diastolic) / 3.0
    derived_map = map_raw * (spo2_effective / 100.0)

    temperature_c = fahrenheit_to_celsius(temperature_f)
    gender_encoded = encode_gender(gender, METADATA["gender_classes"])

    feature_row = {
        "Heart Rate": hr,
        "Hour": now.hour,
        "Minute": now.minute,
        "Second": now.second,
        "Body Temperature": temperature_c,
        "Oxygen Saturation": spo2_raw,
        "Systolic Blood Pressure": systolic,
        "Diastolic Blood Pressure": diastolic,
        "Age": age,
        "Gender_encoded": gender_encoded,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Derived_HRV": hrv,
        "Derived_Pulse_Pressure": pulse_pressure,
        "Derived_BMI": bmi,
        "Derived_MAP": derived_map,
    }

    csv_row = {
        "Timestamp": ts,
        "Heart Rate": hr,
        "Oxygen Saturation": spo2_raw,
        "Status": status,
        "Temperature (F)": temperature_f,
        "Age": age,
        "Gender": gender,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Systolic Blood Pressure": systolic,
        "Diastolic Blood Pressure": diastolic,
        "Derived_HRV": hrv,
        "Derived_Pulse_Pressure": pulse_pressure,
        "Derived_BMI": bmi,
        "Derived_MAP": derived_map,
    }
    return feature_row, csv_row


def predict_risk(feature_row: dict) -> tuple[str, float]:
    feature_cols = METADATA["feature_cols"]
    scale_cols = METADATA["scale_cols"]

    frame = pd.DataFrame([feature_row])[feature_cols]
    frame = frame.astype({col: float for col in scale_cols})
    scaled_values = SCALER.transform(frame[scale_cols])
    scaled_frame = pd.DataFrame(scaled_values, columns=scale_cols, index=frame.index)
    frame.loc[:, scale_cols] = scaled_frame

    prob = RF_MODEL.predict_proba(frame)[0]
    idx = int(np.argmax(prob))
    risk_label = METADATA["risk_classes"][idx]
    confidence = float(np.max(prob))
    return risk_label, confidence


class RealtimeHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/predict":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            raw_len = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(raw_len).decode("utf-8")
            payload = json.loads(raw_body)

            feature_row, csv_row = build_feature_row(payload)
            risk_label, confidence = predict_risk(feature_row)
            append_patients_csv({**csv_row, "Predicted_Risk_Category": risk_label, "Confidence": confidence})

            response = {
                "predicted_risk_category": risk_label,
                "confidence": round(confidence, 4),
                "generated_systolic_bp": feature_row["Systolic Blood Pressure"],
                "generated_diastolic_bp": feature_row["Diastolic Blood Pressure"],
                "derived_hrv": round(feature_row["Derived_HRV"], 3),
                "derived_pulse_pressure": round(feature_row["Derived_Pulse_Pressure"], 3),
                "derived_bmi": round(feature_row["Derived_BMI"], 3),
                "derived_map": round(feature_row["Derived_MAP"], 3),
            }
            self._send_json(200, response)
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8000), RealtimeHandler)
    print("Realtime prediction server running on http://127.0.0.1:8000")
    print("POST /predict | GET /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
