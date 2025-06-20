### Supported Dataset

We support the evaluation using [MoGe]() benchmark and [Marigold]() benchmark.

### Dataset Preparation

#### MoGe Benchmark
Please follow [MoGe](https://github.com/microsoft/MoGe/blob/main/docs/eval.md) to download and unzip the data.


#### Marigold Benchmark
To be done.



### Unified Data Format
- data['scene_name']: string, e.g., 'chess_seq-03'
- data['image']: [3,H,W], 0~255
- data['depth']: [H,W], float
- data['depth_mask']: [H,W], bool