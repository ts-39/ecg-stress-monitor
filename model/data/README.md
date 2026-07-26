# Model Data

This directory is intentionally mostly empty in the repository.
The actual data files are not committed because:
- `WESAD/` is a 16 GB licensed research dataset
- `Dataset/*.json` are generated from WESAD (recreate with `prepare_data.ipynb`)

## How to populate

1. Download WESAD from https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx
2. Place extracted subject folders at `model/data/WESAD/S2/`, `S3/` … `S17/`
3. Run `model/prepare_data.ipynb` to generate `Dataset/WESADECG_S*.json`
4. `Testing/WESADECG_S17.json` is also generated in that notebook (hold-out set)
