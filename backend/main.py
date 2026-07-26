import asyncio
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.requests import Request

from stress_inference import StressPredictor, extract_features
from stress_db import init_db, insert_log, get_today_logs, get_logs_in_range
import calendar_client

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
MODEL_PATH   = Path(__file__).parent / "stress_model_60s.pkl"

# ─── Shared state ─────────────────────────────────────────────────────────────
ecg_pending: deque = deque()
ppi_pending: list  = []
rri_timed:   deque = deque(maxlen=600)   # (timestamp, rri_ms) – ~5 min at 120bpm
connected_websockets: set = set()

ble_status = {
    "connected": False,
    "device_name": None,
    "error": None,
    "scanning": False,
}

calibration = {
    "state":      "idle",   # idle | collecting | done | error
    "start_time": None,
    "duration":   60,
}

calendar_cache = {
    "event":      None,   # current event dict or None
    "fetched_at": 0.0,    # unix timestamp of last fetch
}

predictor: StressPredictor | None = (
    StressPredictor(MODEL_PATH) if MODEL_PATH.exists() else None
)
if predictor is None:
    print("[Stress] stress_model.pkl not found — stress detection disabled")


# ─── BLE callbacks ────────────────────────────────────────────────────────────

def ecg_callback(data):
    ecg_pending.extend(data.data)


def hr_callback(data):
    try:
        now = time.time()
        for rr_1024 in data.rr_intervals:
            rr_ms = int(rr_1024 * 1000 / 1024)
            ppi_pending.append(rr_ms)
            rri_timed.append((now, rr_ms))
    except Exception as e:
        print(f"[HR] callback error: {e}")


# ─── BLE task ─────────────────────────────────────────────────────────────────

async def ble_task():
    from bleak import BleakScanner
    from polar_python import PolarDevice

    if ble_status["connected"] or ble_status["scanning"]:
        return

    ble_status["scanning"] = True
    ble_status["error"] = None
    print("[BLE] Scanning for Polar H10…")

    try:
        devices = await BleakScanner.discover(timeout=10.0, return_adv=True)
    except Exception as e:
        ble_status["error"] = f"Scan failed: {e}"
        ble_status["scanning"] = False
        print(f"[BLE] {ble_status['error']}")
        return

    polar_device = None
    for _, (device, adv) in devices.items():
        if adv.local_name and "Polar H10" in adv.local_name:
            polar_device = device
            break

    if polar_device is None:
        ble_status["error"] = "Polar H10 not found"
        ble_status["scanning"] = False
        print("[BLE] Polar H10 not found.")
        return

    print(f"[BLE] Found: {polar_device.name}")
    ble_status["device_name"] = polar_device.name
    ble_status["scanning"] = False

    try:
        polar = PolarDevice(polar_device)
        await polar.connect()
        ble_status["connected"] = True
        print("[BLE] Connected!")

        await polar.start_ecg_stream(ecg_callback, 130, 14)
        print("[BLE] ECG streaming started.")

        try:
            await polar.start_hr_stream(hr_callback)
            print("[BLE] HR/RR streaming started.")
        except Exception as e:
            print(f"[BLE] HR stream failed (ECG continues): {e}")

        while True:
            await asyncio.sleep(1)

    except Exception as e:
        ble_status["connected"] = False
        ble_status["error"] = f"Disconnected: {e}"
        print(f"[BLE] {ble_status['error']}")


# ─── Broadcaster (ECG + PPI → WebSocket) ─────────────────────────────────────

async def broadcaster():
    while True:
        await asyncio.sleep(0.1)

        if not connected_websockets:
            if len(ecg_pending) > 2000:
                ecg_pending.clear()
            ppi_pending.clear()
            continue

        dead: set = set()

        if ecg_pending:
            batch = list(ecg_pending)
            ecg_pending.clear()
            msg = json.dumps({"type": "ecg", "samples": batch})
            for ws in connected_websockets:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)

        if ppi_pending:
            events = list(ppi_pending)
            ppi_pending.clear()
            msg = json.dumps({"type": "ppi", "values": events})
            for ws in connected_websockets - dead:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)

        for ws in dead:
            connected_websockets.discard(ws)


# ─── Stress inference task ────────────────────────────────────────────────────

async def stress_task():
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(5.0)

        if not connected_websockets:
            continue

        now = time.time()

        # Auto-complete calibration after duration seconds
        if predictor is not None and calibration["state"] == "collecting":
            elapsed = now - calibration["start_time"]
            if elapsed >= calibration["duration"]:
                cal_rri = [rri for ts, rri in rri_timed
                           if ts >= calibration["start_time"]]
                ok = await loop.run_in_executor(None, predictor.set_baseline, cal_rri)
                calibration["state"] = "done" if ok else "error"
                print(f"[Stress] Calibration {'done' if ok else 'failed'} "
                      f"({len(cal_rri)} beats)")

        # Always compute raw HRV features (model / calibration not required)
        window_rri = [rri for ts, rri in rri_timed if ts >= now - 60.0]
        if len(window_rri) < 10:
            continue

        calibrated = predictor is not None and calibration["state"] == "done"

        if not calibrated:
            raw = await loop.run_in_executor(None, extract_features, window_rri)
            if raw is None:
                continue
            feat_msg = json.dumps({
                "type": "features",
                "raw":  [round(float(x), 6) for x in raw],
            })
            dead: set = set()
            for ws in connected_websockets:
                try:
                    await ws.send_text(feat_msg)
                except Exception:
                    dead.add(ws)
            for ws in dead:
                connected_websockets.discard(ws)
            continue

        result = await loop.run_in_executor(None, predictor.predict, window_rri)
        if result is None:
            continue

        # Refresh current calendar event every 60 s
        if calendar_client.is_connected() and now - calendar_cache["fetched_at"] > 60:
            try:
                ev = await loop.run_in_executor(None, calendar_client.get_current_event)
                calendar_cache["event"]      = ev
                calendar_cache["fetched_at"] = now
            except Exception as e:
                print(f"[Calendar] refresh error: {e}")

        current_event = calendar_cache["event"]

        # Persist to SQLite
        raw = result.get("raw", [])
        hr_bpm   = round(60.0 / raw[6], 1) if len(raw) > 6 and raw[6] > 0 else None
        rmssd_ms = round(raw[8] * 1000, 1) if len(raw) > 8 else None
        last_rri = rri_timed[-1][1] if rri_timed else None
        insert_log(
            timestamp=now,
            stress_prob=result["prob"],
            is_stress=result["is_stress"],
            hr_bpm=hr_bpm,
            rri_ms=last_rri,
            rmssd_ms=rmssd_ms,
            event_id=current_event["id"]    if current_event else None,
            event_title=current_event["title"] if current_event else None,
        )

        # Include current event in WebSocket payload
        result["current_event"] = current_event

        msg  = json.dumps({"type": "stress", **result})
        dead: set = set()
        for ws in connected_websockets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            connected_websockets.discard(ws)

        print(f"[Stress] prob={result['prob']:.3f}  stress={result['is_stress']}")


# ─── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    asyncio.create_task(broadcaster())
    asyncio.create_task(stress_task())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status")
async def api_status():
    now = time.time()
    remaining = 0
    if calibration["state"] == "collecting" and calibration["start_time"]:
        remaining = max(0.0, calibration["duration"] - (now - calibration["start_time"]))
    return JSONResponse({
        **ble_status,
        "calibration":           calibration["state"],
        "calibration_remaining": int(remaining),
    })


@app.post("/api/connect")
async def api_connect():
    asyncio.create_task(ble_task())
    return JSONResponse({"ok": True})


@app.get("/api/logs/today")
async def api_logs_today():
    return JSONResponse(get_today_logs())


# ─── Google Calendar endpoints ────────────────────────────────────────────────

@app.get("/api/calendar/status")
async def api_calendar_status():
    loop = asyncio.get_running_loop()
    connected = await loop.run_in_executor(None, calendar_client.is_connected)
    return JSONResponse({
        "connected": connected,
        "current_event": calendar_cache["event"],
    })


@app.post("/api/calendar/connect")
async def api_calendar_connect():
    loop = asyncio.get_running_loop()
    auth_url = await loop.run_in_executor(None, calendar_client.get_auth_url)
    return JSONResponse({"auth_url": auth_url})


@app.get("/api/calendar/callback")
async def api_calendar_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        return JSONResponse({"ok": False, "reason": "no code"}, status_code=400)
    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(None, calendar_client.handle_callback, code)
    if ok:
        # Immediately fetch current event after auth
        try:
            ev = await loop.run_in_executor(None, calendar_client.get_current_event)
            calendar_cache["event"]      = ev
            calendar_cache["fetched_at"] = time.time()
        except Exception:
            pass
    return RedirectResponse(url="/")


@app.get("/api/calendar/today")
async def api_calendar_today():
    loop = asyncio.get_running_loop()
    events = await loop.run_in_executor(None, calendar_client.get_today_events)
    return JSONResponse(events)


@app.get("/api/calendar/current")
async def api_calendar_current():
    loop = asyncio.get_running_loop()
    ev = await loop.run_in_executor(None, calendar_client.get_current_event)
    return JSONResponse(ev)


@app.get("/api/insights/event_stats")
async def api_event_stats(start: float, end: float):
    loop = asyncio.get_running_loop()
    logs = await loop.run_in_executor(None, get_logs_in_range, start, end)
    if not logs:
        return JSONResponse(None)
    probs  = [l["stress_prob"] for l in logs]
    hrs    = [l["hr_bpm"]   for l in logs if l["hr_bpm"]   is not None]
    rmssds = [l["rmssd_ms"] for l in logs if l["rmssd_ms"] is not None]
    trend  = [[l["timestamp"], round(l["stress_prob"] * 100, 1)] for l in logs]
    return JSONResponse({
        "avg_stress":   round(sum(probs) / len(probs) * 100, 1),
        "peak_stress":  round(max(probs) * 100, 1),
        "avg_hr":       round(sum(hrs) / len(hrs), 1)      if hrs    else None,
        "avg_rmssd":    round(sum(rmssds) / len(rmssds), 1) if rmssds else None,
        "sample_count": len(logs),
        "trend":        trend,
    })


@app.post("/api/calibrate/start")
async def api_calibrate_start():
    if not ble_status["connected"]:
        return JSONResponse({"ok": False, "reason": "not connected"}, status_code=400)
    calibration["state"]      = "collecting"
    calibration["start_time"] = time.time()
    print("[Stress] Calibration started")
    return JSONResponse({"ok": True})


@app.websocket("/ws/ecg")
async def ws_ecg(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        connected_websockets.discard(websocket)
