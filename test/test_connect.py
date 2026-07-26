import asyncio
from bleak import BleakClient

ADDRESS = "38BF495F-3AA6-86C0-749F-E19502DAF9CF"


async def main():
    async with BleakClient(ADDRESS) as client:

        print("Connected:", client.is_connected)

        services = client.services

        print("\n=== SERVICES ===")

        for service in services:
            print(service.uuid)

            for char in service.characteristics:
                print("  ", char.uuid)


asyncio.run(main())