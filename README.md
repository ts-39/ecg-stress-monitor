# ECG Stress Monitor

A real-time physiological monitoring system that estimates stress-related states
from ECG-derived heart rate variability (HRV).

The application streams ECG data from a Polar H10 chest strap, extracts HRV
features in real time, estimates stress probability using a LightGBM model
trained on the WESAD dataset, and visualizes the results together with Google
Calendar events to help users understand how daily activities relate to their
physiological responses.

---

# Why I Built This

One of my long-term interests is the future of wearable healthcare.

I believe wearable devices will eventually evolve beyond simply reporting heart
rate or sleep scores. Instead, they will continuously collect physiological
signals and help build a personalized digital representation of our health.

Although we are still far from that future, I wanted to explore one practical
step toward it.

Stress is closely related to many everyday activities such as meetings,
studying, commuting, exercising, and working. However, most people cannot see
how their physiological state changes during those moments.

This project aims to bridge that gap by combining real-time physiological
signals with contextual information from Google Calendar, allowing users to
review not only *how stressed they were*, but also *what they were doing at
that time*.

This application is **not intended for medical diagnosis**.
Instead, it is designed as a research prototype that estimates
stress-related physiological states from ECG-derived HRV.

---

# Tech Stack

| Layer | Stack |
|---|---|
| Hardware | Polar H10 (BLE, 130 Hz ECG) |
| Backend | FastAPI + WebSocket, Python 3.12 |
| ML | LightGBM, 15 HRV features, 60 s RRI window |
| Frontend | Vanilla JS + Canvas API |
| Storage | SQLite |
| Calendar | Google Calendar API v3, OAuth 2.0 + PKCE |

---

# Directory Structure

```text
ecg-stress-monitor/
├── backend/
│   ├── main.py
│   ├── stress_inference.py
│   ├── stress_db.py
│   ├── calendar_client.py
│   ├── stress_model_60s.pkl
│   └── credentials/
├── frontend/
│   └── index.html
├── model/
│   ├── prepare_test.py
│   ├── prepare_data.ipynb
│   ├── lgbm.py
│   ├── model_lgmb.ipynb
│   ├── experiments/
│   └── data/
├── test/
├── requirements-app.txt
└── requirements-model.txt
```

---

# Quick Start

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements-app.txt

uvicorn backend.main:app --reload
```

Open

```
http://localhost:8000
```

and connect a Polar H10 sensor.

---

# Google Calendar Setup

See

```
backend/credentials/README.md
```

for OAuth configuration.

---

# Machine Learning Model

The stress classifier was trained using the public **WESAD** dataset.

During development I experimented with both a neural network and LightGBM.

Although both approaches produced reasonable performance, I ultimately selected
LightGBM for several reasons:

- slightly better validation performance
- lightweight inference suitable for real-time applications
- well suited for tabular HRV features
- easier future integration with explainability methods such as SHAP

Instead of feeding raw ECG directly into the model, the system first extracts
15 HRV features from a rolling 60-second RR-interval window before performing
inference.

This design significantly reduces computational cost while maintaining
interpretable physiological features.

---

# Model Training

## 1. Download WESAD

Paper

https://dl.acm.org/doi/10.1145/3242969.3242985

Dataset

https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx

Place the subject folders in

```
model/data/WESAD/
```

---

## 2. Feature Extraction

Run

```
prepare_data.ipynb
```

to extract HRV features from ECG.

---

## 3. Train

Run

```
model_lgmb.ipynb
```

The resulting model

```
stress_model_60s.pkl
```

should be copied into

```
backend/
```

---

# System Pipeline

```text
Polar H10
      │
      │ ECG / RR intervals
      ▼
FastAPI Backend
      │
      ├── BLE acquisition
      ├── ECG streaming
      ├── HRV feature extraction
      ├── LightGBM inference
      ├── SQLite logging
      └── Google Calendar integration
      ▼
Browser
      │
      ├── ECG waveform
      ├── HRV dashboard
      ├── Stress timeline
      └── Calendar events
```

---

# Calibration

HRV varies significantly between individuals.

Before inference, the application records a **60-second resting baseline**.

Subsequent HRV features are normalized relative to this baseline, allowing the
system to better reflect each user's physiological characteristics rather than
using a single universal reference.

---

# Engineering Challenges

## Training–Serving Consistency

One of the most important challenges during development was ensuring that the
training pipeline matched the real-time inference pipeline.

Initially, the model predicted nearly **100% stress probability even while the
user was completely relaxed**.

Rather than adjusting thresholds, I investigated the entire preprocessing
pipeline.

The issue turned out to be a mismatch between training and deployment:

- Training used the entire neutral section as the baseline.
- The application calibrated using only the first 60 seconds.

After retraining the model using the same 60-second baseline strategy, the
predictions became much more consistent with real physiological behavior.

This experience reinforced the importance of keeping preprocessing identical
between training and deployment.

---

## Real-Time Signal Processing

This application performs several asynchronous tasks simultaneously:

- Bluetooth Low Energy communication
- 130 Hz ECG streaming
- RR interval extraction
- HRV feature computation every 5 seconds
- Machine learning inference
- SQLite logging
- Google Calendar synchronization
- WebSocket streaming

Building a stable pipeline capable of performing all of these operations in
real time was one of the main engineering challenges.

---

# Design Decisions

## Adaptive ECG Scaling

ECG signals often contain large artifacts immediately after connection or when
the chest strap shifts.

To avoid poor visualization, the application ignores the first five seconds
when estimating the display range and dynamically rescales the waveform after
the signal stabilizes.

This significantly improves readability during real-time monitoring.

---

## Personal Calibration

HRV differs substantially across individuals.

Instead of using absolute values alone, the application measures each user's
resting baseline during a 60-second calibration period and compares future
measurements against that personalized reference.

---

## Google Calendar Integration

Stress values alone provide little context.

By integrating Google Calendar, physiological changes can be associated with
daily activities such as meetings, studying, exercise, or commuting.

Rather than simply displaying stress probability, the system helps users
understand *what they were doing* when those physiological changes occurred.

---

# Future Work

Several extensions are planned.

- Better ECG artifact removal
- Signal quality estimation
- Additional physiological sensors (sleep, respiration, activity)
- Personalized models trained on long-term user data
- Explainable AI using SHAP
- Trend analysis over weeks and months
- Digital biomarker research
- LLM-assisted feedback and personalized recommendations based on physiological history

Ultimately, I hope to expand this project toward personalized healthcare
applications that combine wearable sensing, machine learning, and contextual
information to better understand human health.

---

# Model Summary

| Parameter | Value |
|---|---|
| Dataset | WESAD |
| Algorithm | LightGBM |
| Features | 15 HRV features |
| Window | 60 s RRI |
| Inference | Every 5 s |
| Output | Stress probability |