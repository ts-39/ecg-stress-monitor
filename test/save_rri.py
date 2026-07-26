import asyncio
import csv
import time
from datetime import datetime

from bleak import BleakScanner
from polar_python import PolarDevice

rri_rows = []


def hr_callback(data):
    """
    Polar RR interval callback
    """

    now = time.time()

    try:
        for rr_1024 in data.rr_intervals:

            rr_ms = int(rr_1024 * 1000 / 1024)

            rri_rows.append(
                {
                    "timestamp": now,
                    "rri_ms": rr_ms,
                    "hr_bpm": round(60000 / rr_ms, 2),
                }
            )

            print(
                f"RRI={rr_ms:4d} ms "
                f"HR={60000/rr_ms:5.1f} bpm"
            )

    except Exception as e:
        print("callback error:", e)


async def main():

    print("Scanning Polar H10...")

    devices = await BleakScanner.discover(
        timeout=10.0,
        return_adv=True,
    )

    polar_device = None

    for _, (device, adv) in devices.items():

        if adv.local_name and "Polar H10" in adv.local_name:
            polar_device = device
            break

    if polar_device is None:
        raise RuntimeError("Polar H10 not found")

    print("Found:", polar_device)

    polar = PolarDevice(polar_device)

    await polar.connect()

    print("Connected")

    await polar.start_hr_stream(hr_callback)

    print("Recording...")
    print("Press Ctrl+C to stop")

    try:

        while True:
            await asyncio.sleep(1)

    finally:

        filename = (
            "polar_rri_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv"
        )

        with open(filename, "w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "rri_ms",
                    "hr_bpm",
                ],
            )

            writer.writeheader()
            writer.writerows(rri_rows)

        print()
        print("Saved:", filename)
        print("Rows :", len(rri_rows))

        try:
            await polar.disconnect()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())