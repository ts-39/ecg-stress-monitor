import asyncio
from bleak import BleakScanner

async def main():

    devices = await BleakScanner.discover(
        timeout=20.0,
        return_adv=True
    )

    for address, (device, adv) in devices.items():

        print("DEVICE")
        print("name:", device.name)
        print("address:", device.address)
        print("local_name:", adv.local_name)
        print("uuids:", adv.service_uuids)
        print()

asyncio.run(main())