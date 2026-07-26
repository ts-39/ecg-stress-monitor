# ECG Stress Monitor

Real-time stress detection using a Polar H10 ECG sensor.
HRV features are extracted from RR-intervals every 5 seconds and fed into a
LightGBM classifier trained on the WESAD dataset.
Google Calendar integration overlays your schedule on a day-long stress timeline
so you can review how each event affected your physiology.

---

## Tech stack

| Layer | Stack |
|---|---|
| Hardware | Polar H10 (BLE, 130 Hz ECG) |
| Backend | FastAPI + WebSocket, Python 3.12 |
| ML | LightGBM, 15 HRV features, 60 s RRI window |
| Frontend | Vanilla JS + Canvas API (single HTML file) |
| Storage | SQLite (`stress_log.db`) |
| Calendar | Google Calendar API v3, OAuth 2.0 + PKCE |

---

## Directory structure

```
ecg-stress-monitor/
├── backend/              FastAPI server, BLE, ML inference, SQLite, Calendar
│   ├── main.py
│   ├── stress_inference.py
│   ├── stress_db.py
│   ├── calendar_client.py
│   ├── stress_model_60s.pkl   production LightGBM model
│   └── credentials/           OAuth secrets — NOT committed (see below)
├── frontend/
│   └── index.html        Full UI (ECG canvas, HRV panel, stress timeline)
├── model/                Model training code (uses WESAD dataset)
│   ├── prepare_test.py   Extract HRV features from raw WESAD ECG
│   ├── prepare_data.ipynb
│   ├── lgbm.py           LightGBM training script
│   ├── model_lgmb.ipynb  Training notebook → produces stress_model_60s.pkl
│   ├── experiments/      Old PyTorch experiments (not used in production)
│   └── data/
│       ├── Dataset/      Preprocessed JSON files (generated — see below)
│       └── Testing/      Hold-out subject (S17)
├── test/                 BLE / ECG integration tests
├── requirements-app.txt  Dependencies for running the app
└── requirements-model.txt  Dependencies for model training
```

---

## Quick start (app)

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements-app.txt

# 2. Add Google Calendar credentials (optional — see backend/credentials/README.md)

# 3. Run
uvicorn backend.main:app --reload

# 4. Open http://localhost:8000 in your browser
# 5. Click "Connect to Polar H10" (make sure the sensor is on and nearby)
```

---

## Google Calendar setup

See [`backend/credentials/README.md`](backend/credentials/README.md) for step-by-step instructions.

---

## Model training

### 1. Get the WESAD dataset

WESAD is a public research dataset from the University of Bonn / Fraunhofer IAIS.

- Paper: https://dl.acm.org/doi/10.1145/3242969.3242985
- Download: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx (requires registration)

After downloading, extract and place the subject folders at:

```
model/data/WESAD/S2/
model/data/WESAD/S3/
...
model/data/WESAD/S17/
```

### 2. Preprocess

Run `model/prepare_data.ipynb` (or `model/prepare_test.py`) to extract HRV features.
This generates `model/data/Dataset/WESADECG_S*.json` files.

### 3. Train

Run `model/model_lgmb.ipynb` end-to-end.
The final cell saves `stress_model_60s.pkl`.
Copy it to `backend/stress_model_60s.pkl` to use it in the app.

---

## How it works

```
Polar H10 (BLE)
    │ ECG 130 Hz / RR-intervals
    ▼
FastAPI backend
    │ broadcaster()  — streams ECG + RRI to browser via WebSocket (100 ms)
    │ stress_task()  — computes 15 HRV features every 5 s
    │   if calibrated → LightGBM predict → SQLite log → send stress msg
    │   else          → send raw features only
    ▼
Browser (index.html)
    Canvas ECG waveform | HRV features panel | Stress timeline
    ▲
    Google Calendar API  — event zones overlaid on timeline
    SQLite               — today's history loaded on startup
```

### Calibration

Press "Start Calibration" while relaxed.
The app records 60 s of HRV as your personal baseline.
All subsequent feature vectors are divided by this baseline before
being passed to the model, correcting for individual physiological differences.

---

## Model details

| Parameter | Value |
|---|---|
| Dataset | WESAD (15 subjects for training, S17 held out) |
| Algorithm | LightGBM (binary classification) |
| Features | 15 HRV features from a 60 s RRI window |
| Threshold | 0.35 (stress probability) |
| Training samples | 2,567 |
