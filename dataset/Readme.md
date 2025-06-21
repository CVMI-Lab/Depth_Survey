### Supported Dataset

We support the evaluation using [MoGe]() benchmark and [Marigold]() benchmark.

### Dataset Preparation

#### MoGe Benchmark
Please follow [MoGe](https://github.com/microsoft/MoGe/blob/main/docs/eval.md) to download and unzip the data.

**Troubleshoot**

Pok#U00e9mon_Omega_Ruby_Alpha_Sapphire_Dual_Pack_Nintendo_3DS
- If you are stucked when processing `GSO` dataset, please consider replace `é` in `$Datadir$/GSO/.index.txt` with `#U00e9`. E.g., `Pokémon_Omega_Ruby_Alpha_Sapphire_Dual_Pack_Nintendo_3DS` -> `Pok#U00e9mon_Omega_Ruby_Alpha_Sapphire_Dual_Pack_Nintendo_3DS`

You can use [`gen_config.py`](../configs/moge_benchmark/gen_config.py) to generate config yamls of different datasets for one method conveniently.

#### Marigold Benchmark
To be done.


### Unified Data Format
- data['scene_name']: string, e.g., 'chess_seq-03'
- data['image']: [3,H,W], 0~255
- data['depth']: [H,W], float
- data['depth_mask']: [H,W], bool

