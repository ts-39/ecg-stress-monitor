import asyncio
from bleak import BleakClient, BleakScanner

PMD_SERVICE = "fb005c80-02e7-f387-1cad-8acd2d8df0c8"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"


def notification_handler(sender, data):
    print("ECG packet:", len(data), "bytes")


async def main():

    device = await BleakScanner.find_device_by_filter(
        lambda d, ad:
            ad.local_name is not None
            and "Polar H10" in ad.local_name
    )

    if not device:
        raise RuntimeError("Polar H10 not found")

    async with BleakClient(device) as client:

        print("Connected")

        await client.start_notify(
            PMD_DATA,
            notification_handler
        )

        # ECG開始コマンド
        ecg_start = bytearray([
            0x02,
            0x00,
            0x00,
            0x01,
            0x82,
            0x00,
            0x01,
            0x01,
            0x0E,
            0x00
        ])

        await client.write_gatt_char(
            PMD_CONTROL,
            ecg_start,
            response=True
        )

        print("ECG started")

        while True:
            await asyncio.sleep(1)


asyncio.run(main())