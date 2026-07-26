import asyncio
import time

from bleak import BleakScanner
from polar_python import PolarDevice

all_samples = []
packet_count = 0
start_time = time.time()


def ecg_callback(data):
    global packet_count

    packet_count += 1

    # ECGサンプルを蓄積
    all_samples.extend(data.data)

    # 10パケットごとに状況表示
    if packet_count % 10 == 0:
        elapsed = time.time() - start_time

        print(
            f"packets={packet_count} "
            f"samples={len(all_samples)} "
            f"elapsed={elapsed:.1f}s"
        )


async def main():

    devices = await BleakScanner.discover(
        timeout=10.0,
        return_adv=True
    )

    polar_device = None

    for _, (device, adv) in devices.items():

        if (
            adv.local_name
            and "Polar H10" in adv.local_name
        ):
            polar_device = device
            break

    if polar_device is None:
        raise RuntimeError("Polar H10 not found")

    print("Using:", polar_device)

    polar = PolarDevice(polar_device)

    await polar.connect()

    print("Connected")

    await polar.start_ecg_stream(
        ecg_callback,
        130,
        14
    )

    print("Streaming...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nStopped")

        print(
            f"Total packets: {packet_count}"
        )

        print(
            f"Total samples: {len(all_samples)}"
        )

        print(
            f"First 20 samples:"
        )

        print(all_samples[:20])

        await polar.disconnect()


if __name__ == "__main__":
    asyncio.run(main())