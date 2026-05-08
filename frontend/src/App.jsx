import { useEffect, useMemo, useState } from "react";
import { database, firebaseInitError } from "./firebase";
import { onValue, ref } from "firebase/database";
import {
  CategoryScale,
  Chart as ChartJS,
  LineElement,
  LinearScale,
  PointElement,
} from "chart.js";
import { Line } from "react-chartjs-2";
import "./App.css";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement);

const WINDOW_SIZE = 10;
const API_BASE = "http://127.0.0.1:8000";
const NLP_API_BASE = "http://127.0.0.1:8001";

const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;

const std = (arr) => {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length);
};

function App() {
  const [heartRate, setHeartRate] = useState("--");
  const [spo2, setSpo2] = useState("--");
  const [temperature, setTemperature] = useState("--");
  const [status, setStatus] = useState("--");
  const [timeLabel, setTimeLabel] = useState("--");

  const [labels, setLabels] = useState([]);
  const [hrData, setHrData] = useState([]);
  const [spo2Data, setSpo2Data] = useState([]);
  const [tempData, setTempData] = useState([]);
  const [dataset, setDataset] = useState([]);

  const [age, setAge] = useState("30");
  const [gender, setGender] = useState("Male");
  const [weight, setWeight] = useState("70");
  const [height, setHeight] = useState("1.70");

  const [prediction, setPrediction] = useState("Waiting...");
  const [confidence, setConfidence] = useState("--");
  const [derivedInfo, setDerivedInfo] = useState(null);
  const [apiStatus, setApiStatus] = useState("Not connected");
  const [firebaseStatus, setFirebaseStatus] = useState("Connecting...");
  const [physiologicalScore, setPhysiologicalScore] = useState("0.5");
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [nlpScore, setNlpScore] = useState("--");
  const [finalScore, setFinalScore] = useState("--");
  const [finalRiskCategory, setFinalRiskCategory] = useState("--");
  const [nlpConfidence, setNlpConfidence] = useState("--");
  const [nlpApiStatus, setNlpApiStatus] = useState("Checking...");
  const [reportSummary, setReportSummary] = useState("");
  const [analysisSource, setAnalysisSource] = useState("--");

  const patientProfileReady = useMemo(() => {
    return Number(age) > 0 && Number(weight) > 0 && Number(height) > 0 && gender;
  }, [age, weight, height, gender]);

  const extractSensorPayload = (raw) => {
    if (!raw || typeof raw !== "object") {
      return null;
    }

    // Some Firebase layouts store one extra nested node: hospital/patient1/patient1
    const base = raw.patient1 && typeof raw.patient1 === "object" ? raw.patient1 : raw;

    const hr =
      base.heartRate_BPM ??
      base.heart_rate_BPM ??
      base.heartRate ??
      base.hr ??
      null;
    const spo2 = base.spo2_percent ?? base.SpO2 ?? base.spo2 ?? null;
    const temperature =
      base.temperature_F ??
      base.temperature_F_ ??
      base.temperatureF ??
      base.temp_F ??
      null;
    const sensorStatus = base.status ?? base.Status ?? "UNKNOWN";

    if (hr == null || spo2 == null || temperature == null) {
      return null;
    }

    return {
      heartRate: Number(hr),
      spo2: Number(spo2),
      temperature: Number(temperature),
      status: String(sensorStatus),
    };
  };

  const callPredictionApi = async (sensor) => {
    if (!patientProfileReady) {
      setApiStatus("Enter age, gender, weight, and height.");
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patientId: "patient1",
          heartRate_BPM: sensor.heartRate,
          spo2_percent: sensor.spo2,
          temperature_F: sensor.temperature,
          status: sensor.status,
          age: Number(age),
          gender,
          weight_kg: Number(weight),
          height_m: Number(height),
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || "Prediction failed");
      }

      const data = await res.json();
      setPrediction(data.predicted_risk_category);
      setConfidence((data.confidence * 100).toFixed(2));
      setDerivedInfo(data);
      setApiStatus("Connected");
    } catch (error) {
      setApiStatus(`API error: ${error.message}`);
    }
  };

  useEffect(() => {
    if (!database) {
      setFirebaseStatus(
        `${firebaseInitError || "Firebase is not configured."} Running in mock mode.`
      );

      let current = { heartRate: 78, spo2: 98, temperature: 98.6, status: "NORMAL" };
      const pushMockReading = () => {
        current = {
          heartRate: Math.max(55, Math.min(130, current.heartRate + (Math.random() * 8 - 4))),
          spo2: Math.max(90, Math.min(100, current.spo2 + (Math.random() * 2 - 1))),
          temperature: Math.max(96.0, Math.min(102.0, current.temperature + (Math.random() * 0.6 - 0.3))),
          status: "NORMAL",
        };

        const nowLabel = new Date().toLocaleTimeString();
        const rounded = {
          heartRate: Number(current.heartRate.toFixed(0)),
          spo2: Number(current.spo2.toFixed(0)),
          temperature: Number(current.temperature.toFixed(1)),
          status: current.status,
        };

        setHeartRate(rounded.heartRate);
        setSpo2(rounded.spo2);
        setTemperature(rounded.temperature);
        setStatus(rounded.status);
        setTimeLabel(nowLabel);
        setLabels((prevLabels) => [...prevLabels.slice(-39), nowLabel]);
        setHrData((prevHr) => [...prevHr.slice(-39), rounded.heartRate]);
        setSpo2Data((prevSpo2) => [...prevSpo2.slice(-39), rounded.spo2]);
        setTempData((prevTemp) => [...prevTemp.slice(-39), rounded.temperature]);
        callPredictionApi(rounded);
      };

      pushMockReading();
      const timer = setInterval(pushMockReading, 3000);
      return () => clearInterval(timer);
    }

    const patientRef = ref(database, "hospital/patient1");
    const unsubscribe = onValue(
      patientRef,
      (snapshot) => {
      const sensor = extractSensorPayload(snapshot.val());
      if (!sensor) {
        setFirebaseStatus("Connected, but expected sensor keys were not found.");
        return;
      }
      setFirebaseStatus("Connected");
      const nowLabel = new Date().toLocaleTimeString();

      setHeartRate(sensor.heartRate);
      setSpo2(sensor.spo2);
      setTemperature(sensor.temperature);
      setStatus(sensor.status);
      setTimeLabel(nowLabel);

      setLabels((prev) => [...prev.slice(-39), nowLabel]);
      setHrData((prev) => [...prev.slice(-39), sensor.heartRate]);
      setSpo2Data((prev) => [...prev.slice(-39), sensor.spo2]);
      setTempData((prev) => [...prev.slice(-39), sensor.temperature]);

      callPredictionApi({
        heartRate: sensor.heartRate,
        spo2: sensor.spo2,
        temperature: sensor.temperature,
        status: sensor.status,
      });
      },
      (error) => {
        setFirebaseStatus(`Firebase error: ${error.message}`);
      }
    );

    return () => unsubscribe();
  }, [age, gender, height, weight, patientProfileReady, database, firebaseInitError]);

  useEffect(() => {
    if (hrData.length < WINDOW_SIZE) {
      setDataset([]);
      return;
    }

    const features = [];
    for (let i = 0; i <= hrData.length - WINDOW_SIZE; i += 1) {
      const hrWindow = hrData.slice(i, i + WINDOW_SIZE);
      const spo2Window = spo2Data.slice(i, i + WINDOW_SIZE);
      const tempWindow = tempData.slice(i, i + WINDOW_SIZE);

      features.push({
        window_start: labels[i],
        window_end: labels[i + WINDOW_SIZE - 1],
        hr_mean: mean(hrWindow).toFixed(2),
        hr_std: std(hrWindow).toFixed(2),
        spo2_mean: mean(spo2Window).toFixed(2),
        temp_mean: mean(tempWindow).toFixed(2),
        temp_std: std(tempWindow).toFixed(2),
      });
    }
    setDataset(features);
  }, [hrData, spo2Data, tempData, labels]);

  useEffect(() => {
    let cancelled = false;

    const checkNlpHealth = async () => {
      try {
        const res = await fetch(`${NLP_API_BASE}/health`);
        if (!res.ok) {
          throw new Error(`Health check failed (${res.status})`);
        }
        if (!cancelled) {
          setNlpApiStatus("Connected");
        }
      } catch (error) {
        if (!cancelled) {
          setNlpApiStatus(`Not connected: ${error.message}`);
        }
      }
    };

    checkNlpHealth();
    const intervalId = setInterval(checkNlpHealth, 30000);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, []);

  const downloadCSV = () => {
    if (dataset.length === 0) {
      return;
    }
    const header = Object.keys(dataset[0]).join(",");
    const rows = dataset.map((row) => Object.values(row).join(",")).join("\n");
    const csv = `${header}\n${rows}`;
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "patients_sliding_window.csv";
    a.click();
  };

  const heartRateChart = {
    labels,
    datasets: [{ label: "Heart Rate (BPM)", data: hrData, borderColor: "#d62828", tension: 0.35 }],
  };

  const spo2Chart = {
    labels,
    datasets: [{ label: "SpO2 (%)", data: spo2Data, borderColor: "#1d4ed8", tension: 0.35 }],
  };

  const handlePdfOnlyNlp = async () => {
    if (!selectedPdf) {
      setNlpApiStatus("Select a PDF first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", selectedPdf);

    try {
      const res = await fetch(`${NLP_API_BASE}/upload-pdf`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "NLP upload failed.");
      }
      const data = await res.json();
      setNlpScore(Number(data.nlp_score).toFixed(4));
      setNlpConfidence((Number(data.confidence) * 100).toFixed(2));
      setReportSummary(data.report_summary || "");
      setAnalysisSource(data.analysis_source || "--");
      setNlpApiStatus("Connected");
    } catch (error) {
      setNlpApiStatus(`NLP API error: ${error.message}`);
    }
  };

  const handlePredictFinalRisk = async () => {
    if (!selectedPdf) {
      setNlpApiStatus("Select a PDF first.");
      return;
    }
    const score = Number(physiologicalScore);
    if (Number.isNaN(score) || score < 0 || score > 1) {
      setNlpApiStatus("Physiological score must be between 0 and 1.");
      return;
    }

    const formData = new FormData();
    formData.append("physiological_score", String(score));
    formData.append("file", selectedPdf);

    try {
      const res = await fetch(`${NLP_API_BASE}/predict-risk`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Final risk prediction failed.");
      }
      const data = await res.json();
      setNlpScore(Number(data.nlp_score).toFixed(4));
      setFinalScore(Number(data.final_score).toFixed(4));
      setFinalRiskCategory(data.risk_category);
      setNlpConfidence(Number(data.confidence).toFixed(2));
      setReportSummary(data.report_summary || "");
      setAnalysisSource(data.analysis_source || "--");
      setNlpApiStatus("Connected");
    } catch (error) {
      setNlpApiStatus(`NLP API error: ${error.message}`);
    }
  };

  return (
    <div className="dashboard">
      <h1 className="hero-title">AI-Powered Smart Health Monitoring</h1>

      <div className="card-grid">
        <div className="metric-card">
          <h3>Heart Rate</h3>
          <p>{heartRate} BPM</p>
        </div>
        <div className="metric-card">
          <h3>SpO2</h3>
          <p>{spo2} %</p>
        </div>
        <div className="metric-card">
          <h3>Temperature</h3>
          <p>{temperature} F</p>
        </div>
        <div className="metric-card">
          <h3>Sensor Status</h3>
          <p>{status}</p>
        </div>
      </div>

      <div className="profile-box">
        <h2>Patient Profile (Required for Prediction)</h2>
        <div className="profile-grid">
          <label>
            Age
            <input type="number" min="1" value={age} onChange={(e) => setAge(e.target.value)} />
          </label>
          <label>
            Gender
            <select value={gender} onChange={(e) => setGender(e.target.value)}>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </label>
          <label>
            Weight (kg)
            <input
              type="number"
              min="1"
              step="0.1"
              value={weight}
              onChange={(e) => setWeight(e.target.value)}
            />
          </label>
          <label>
            Height (m)
            <input
              type="number"
              min="0.5"
              step="0.01"
              value={height}
              onChange={(e) => setHeight(e.target.value)}
            />
          </label>
        </div>
      </div>

      <div className="prediction-box">
        <h2>Predicted Risk Category</h2>
        <p className={prediction === "High Risk" ? "risk-high" : "risk-low"}>{prediction}</p>
        <p>Confidence: {confidence} %</p>
        <p>API: {apiStatus}</p>
        <p>Firebase: {firebaseStatus}</p>
        <p>Last Update: {timeLabel}</p>
        {derivedInfo && (
          <div className="derived-box">
            <p>
              Generated BP: {derivedInfo.generated_systolic_bp}/{derivedInfo.generated_diastolic_bp}
            </p>
            <p>Derived HRV: {derivedInfo.derived_hrv}</p>
            <p>Derived Pulse Pressure: {derivedInfo.derived_pulse_pressure}</p>
            <p>Derived BMI: {derivedInfo.derived_bmi}</p>
            <p>Derived MAP: {derivedInfo.derived_map}</p>
          </div>
        )}
      </div>

      <div className="prediction-box">
        <h2>NLP + Final Risk Scoring</h2>
        <div className="profile-grid">
          <label>
            Physiological Score (0-1)
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={physiologicalScore}
              onChange={(e) => setPhysiologicalScore(e.target.value)}
            />
          </label>
          <label>
            Upload Medical PDF
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setSelectedPdf(e.target.files?.[0] || null)}
            />
          </label>
        </div>
        <div className="button-row">
          <button className="download-btn" onClick={handlePdfOnlyNlp}>
            Compute NLP Score
          </button>
          <button className="download-btn" onClick={handlePredictFinalRisk}>
            Compute Final Risk
          </button>
        </div>
        <div className="card-grid">
          <div className="metric-card">
            <h3>NLP Risk Score</h3>
            <p>{nlpScore}</p>
          </div>
          <div className="metric-card">
            <h3>Physiological Score</h3>
            <p>{physiologicalScore}</p>
          </div>
          <div className="metric-card">
            <h3>Final Score</h3>
            <p>{finalScore}</p>
          </div>
          <div className="metric-card">
            <h3>Risk Category</h3>
            <p>{finalRiskCategory}</p>
          </div>
        </div>
        <p>NLP Confidence: {nlpConfidence} %</p>
        <p>NLP API: {nlpApiStatus}</p>
        <p>Analysis Source: {analysisSource}</p>
        {reportSummary && <p>Report Summary: {reportSummary}</p>}
      </div>

      <button className="download-btn" onClick={downloadCSV}>
        Download Sliding-Window CSV
      </button>

      <div className="chart-wrapper">
        <h3>Heart Rate Trend</h3>
        <Line data={heartRateChart} />
      </div>

      <div className="chart-wrapper">
        <h3>SpO2 Trend</h3>
        <Line data={spo2Chart} />
      </div>

      <div className="table-box">
        <h3>Sliding Window Dataset (Window Size = 10)</h3>
        <table>
          <thead>
            <tr>
              <th>Start</th>
              <th>End</th>
              <th>HR Mean</th>
              <th>HR Std</th>
              <th>SpO2 Mean</th>
              <th>Temp Mean</th>
              <th>Temp Std</th>
            </tr>
          </thead>
          <tbody>
            {dataset.map((row, idx) => (
              <tr key={`${row.window_start}-${idx}`}>
                <td>{row.window_start}</td>
                <td>{row.window_end}</td>
                <td>{row.hr_mean}</td>
                <td>{row.hr_std}</td>
                <td>{row.spo2_mean}</td>
                <td>{row.temp_mean}</td>
                <td>{row.temp_std}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;
